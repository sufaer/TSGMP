import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from resnet import resnet18
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import copy


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*2, embed_dim)
        )
        # 新增门控残差
        self.gate = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, query, key, value):
        attn_out, _ = self.attn(query, key, value)
        # 动态调节注意力贡献
        gate = self.gate(torch.cat([query, attn_out], dim=-1))
        x = query + gate * attn_out
        return self.norm(x + self.ffn(x))

class HybridEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        # 骨干网络初始化（保持原样）
        self.backbone = resnet18(sample_input_D=64, sample_input_H=128, sample_input_W=128, num_seg_classes=1)
        self.backbone.conv_seg = nn.Identity()
        self.backbone.load_state_dict(torch.load("/home/zhoutz/MLDRL/pretrain/resnet_18_23dataset.pth"), strict=False)

        # 共享层
        self.shared_layers = nn.Sequential(
            self.backbone.conv1, self.backbone.bn1, self.backbone.relu, 
            self.backbone.maxpool, self.backbone.layer1, self.backbone.layer2
        )
        
        # 独立模态特征提取
        self.dce_layers = nn.Sequential(
            copy.deepcopy(self.backbone.layer3),
            copy.deepcopy(self.backbone.layer4)
        )
        self.dwi_layers = nn.Sequential(
            copy.deepcopy(self.backbone.layer3),
            copy.deepcopy(self.backbone.layer4)
        )
        
        # 多模态融合
        self.cross_attn = CrossAttention(embed_dim=512, num_heads=8)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def extract_features(self, x, modality):
        x = self.shared_layers(x)
        x = self.dce_layers(x) if modality == 'dce' else self.dwi_layers(x)
        return self.global_pool(x).flatten(1)

    def forward(self, pre_dce, pre_dwi, post_dce, post_dwi):
        # 治疗前特征
        pre_dce_feat = self.extract_features(pre_dce, 'dce')
        pre_dwi_feat = self.extract_features(pre_dwi, 'dwi')
        
        # 治疗后特征
        post_dce_feat = self.extract_features(post_dce, 'dce') 
        post_dwi_feat = self.extract_features(post_dwi, 'dwi')
        
        # 交叉注意力融合
        dce_attn = self.cross_attn(
            pre_dce_feat.unsqueeze(1),
            pre_dwi_feat.unsqueeze(1),
            pre_dwi_feat.unsqueeze(1)
        ).squeeze(1)
        
        dwi_attn = self.cross_attn(
            pre_dwi_feat.unsqueeze(1),
            pre_dce_feat.unsqueeze(1), 
            pre_dce_feat.unsqueeze(1)
        ).squeeze(1)
        
        # 返回治疗前双模态特征 + 治疗后特征（用于辅助监督）
        return {
            'pre_dce': pre_dce_feat,
            'pre_dwi': pre_dwi_feat,
            'post_dce': post_dce_feat,
            'post_dwi': post_dwi_feat,
            'pre_fused': torch.cat([dce_attn, dwi_attn], dim=1)
        }

class FullModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.encoder = HybridEncoder(device)
        
        # 增强版分类器
        self.pcr_classifier = nn.Sequential(
            nn.Linear(2048, 1024),  # pre_fused + post_fused
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # 双模态delta预测
        self.delta_predictor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024)
        )
        
        self.loss_weights = {'cls': 1.0, 'delta': 0.5, 'contrast': 0.3}

    def contrastive_loss(self, pre, post, label, margin=0.7):
        sim = F.cosine_similarity(pre, post)
        return torch.where(
            label == 1,
            F.relu(sim - (1 - margin)),
            F.relu(margin - sim)
        ).mean()

    def forward(self, pre_dce, pre_dwi, post_dce, post_dwi, pcr_label):
        features = self.encoder(pre_dce, pre_dwi, post_dce, post_dwi)
        
        # 治疗后特征融合
        post_fused = torch.cat([
            self.encoder.cross_attn(
                features['post_dce'].unsqueeze(1),
                features['post_dwi'].unsqueeze(1),
                features['post_dwi'].unsqueeze(1)
            ).squeeze(1),
            features['post_dwi']
        ], dim=1)
        
        # 主分类任务
        combined_feat = torch.cat([features['pre_fused'], post_fused], dim=1)
        pcr_pred = self.pcr_classifier(combined_feat)
        
        # 双模态delta预测
        delta_pred = self.delta_predictor(
            torch.cat([features['pre_dce'], features['pre_dwi']], dim=1)
        )
        delta_loss = F.mse_loss(
            delta_pred,
            torch.cat([
                features['post_dce'] - features['pre_dce'],
                features['post_dwi'] - features['pre_dwi']
            ], dim=1)
        )
        
        # 对比学习
        dce_contrast = self.contrastive_loss(
            features['pre_dce'], features['post_dce'], pcr_label
        )
        dwi_contrast = self.contrastive_loss(
            features['pre_dwi'], features['post_dwi'], pcr_label
        )
        contrast_loss = (dce_contrast + dwi_contrast) / 2
        
        # 加权损失
        cls_loss = F.binary_cross_entropy(
            pcr_pred, pcr_label.unsqueeze(1),
            weight=torch.tensor([1.0, 8.0], device=device)[pcr_label.long()].unsqueeze(1)
        )
        total_loss = (self.loss_weights['cls'] * cls_loss + 
                     self.loss_weights['delta'] * delta_loss +
                     self.loss_weights['contrast'] * contrast_loss)
        
        return cls_loss, total_loss    


class FineTuningModel(nn.Module):
    def __init__(self, encoder, clinical_features, experiment_type, device):
        super().__init__()
        self.encoder = encoder

        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # 双模态特征处理
        self.dce_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.dwi_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3) 
        )
        
        # 多模态融合
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256 + 32, 512),  # DCE+DWI+Clinical
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 原型记忆库
        self.proto_pos = nn.Parameter(torch.randn(256), requires_grad=True)
        self.proto_neg = nn.Parameter(torch.randn(256), requires_grad=True)

    def forward(self, pre_dce, pre_dwi, clinical_data):
        # 提取双模态特征（冻结）
        with torch.no_grad():
            enc_output = self.encoder(pre_dce, pre_dwi, pre_dce, pre_dwi)  # 伪治疗后输入
            
        # 投影到统一空间
        dce_feat = self.dce_proj(enc_output['pre_dce'])
        dwi_feat = self.dwi_proj(enc_output['pre_dwi'])
        clinical_feat = self.clinical_net(clinical_data)
        
        # 多模态融合
        fused = self.fusion(torch.cat([dce_feat, dwi_feat, clinical_feat], dim=1))
        return fused, self.classifier(fused)

    def loss(self, fused_feat, pcr_pred, pcr_label):
        # Focal Loss
        focal_loss = F.binary_cross_entropy_with_logits(
            pcr_pred.squeeze(1), pcr_label.float(),
            pos_weight=torch.tensor(5.0, device=pcr_label.device)
        )
        
        # 原型对比损失
        pos_dist = torch.norm(fused_feat - self.proto_pos, dim=1)
        neg_dist = torch.norm(fused_feat - self.proto_neg, dim=1)
        proto_loss = torch.where(
            pcr_label == 1,
            pos_dist - neg_dist + 0.5,  # 正样本应靠近pos原型
            neg_dist - pos_dist + 0.5   # 负样本应靠近neg原型
        ).mean()
        
        # 多样性正则化
        var_loss = -torch.var(fused_feat, dim=0).mean()  # 防止特征坍缩
        
        return focal_loss + 0.3*proto_loss + 0.1*var_loss

