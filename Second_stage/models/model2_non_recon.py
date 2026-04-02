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
    def __init__(self, embed_dim, num_heads=2, dropout = 0.3):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim*2,embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x_query, x_key, x_value):
        attn_output = self.attn(x_query, x_key, x_value, need_weights=False)[0]
        gate_value = self.gate(torch.cat([x_query, attn_output], dim=-1))
        x = x_query + gate_value * attn_output
        x = self.norm1(x)
        x = self.norm2(x + self.mlp(x))
        return x


# 第一阶段：多模态编码器
class HybridEncoder(nn.Module):
    def __init__(self, device):
        super(HybridEncoder, self).__init__()
        self.backbone = resnet18(sample_input_D=64, sample_input_H=128, sample_input_W=128, num_seg_classes=1)
        self.backbone.conv_seg = nn.Identity()  # 去除原模型中不需要的conv_seg部分
        # 加载预训练权重
        pretrained_weights = torch.load("/home/zhoutz/MLDRL/pretrain/resnet_18_23dataset.pth")
        self.backbone.load_state_dict(pretrained_weights, strict=False)

        # 共享部分
        self.shared_layers = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2
        )

        # **专用部分（DCE & DWI 各自的高级特征提取层）**
        self.dce_layer3 = copy.deepcopy(self.backbone.layer3)
        self.dce_layer4 = copy.deepcopy(self.backbone.layer4)

        self.dwi_layer3 = copy.deepcopy(self.backbone.layer3)  # 复制结构
        self.dwi_layer4 = copy.deepcopy(self.backbone.layer4)  # 复制结构

        self.pcr_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn. LayerNorm(256)
        )
        self.contrast_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # **自适应池化**
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # **交叉注意力模块**
        self.cross_attn = CrossAttention(embed_dim = 512, num_heads = 2)

        self.device = device
        self.to(device)

    def extract_features(self, x, layer3, layer4):
        x = self.shared_layers(x)
        x = layer3(x)
        x = layer4(x)
        x = self.global_pool(x).flatten(start_dim=1)
        return x
    
    def encoder_c(self, x):
        x = self.extract_features(x, self.dce_layer3, self.dce_layer4)
        return x

    def encoder_dwi(self, x):
        x = self.extract_features(x, self.dwi_layer3, self.dwi_layer4)
        return x

    def forward(self, pre_dce, pre_dwi, post_dce, post_dwi):

        # 治疗前特征提取
        pre_dce_base = self.encoder_c(pre_dce)
        pre_dwi_base = self.encoder_dwi(pre_dwi)
        # pCR专用特征
        pre_dce_pcr = self.pcr_proj(pre_dce_base)
        pre_dwi_pcr = self.pcr_proj(pre_dwi_base)
        
        # 治疗后特征提取
        post_dce_base = self.encoder_c(post_dce)
        post_dwi_base = self.encoder_dwi(post_dwi)
        # pCR专用特征
        post_dce_pcr = self.pcr_proj(post_dce_base)
        post_dwi_pcr = self.pcr_proj(post_dwi_base)
        # 对比
        post_dce_contrast = self.contrast_proj(post_dce_base)
        post_dwi_contrast = self.contrast_proj(post_dwi_base)

        # 交叉注意力
        pre_dce_attn = self.cross_attn(
            pre_dce_base.unsqueeze(1), 
            pre_dwi_base.unsqueeze(1), 
            pre_dwi_base.unsqueeze(1)).squeeze(1)
        pre_dwi_attn = self.cross_attn(
            pre_dwi_base.unsqueeze(1),
            pre_dce_base.unsqueeze(1),
            pre_dce_base.unsqueeze(1)).squeeze(1)

        pre_pcr = torch.cat([pre_dce_pcr, pre_dwi_pcr], dim=1)  # [B, 512]
        post_pcr = torch.cat([post_dce_pcr, post_dwi_pcr], dim=1)  # [B, 512]
        # 对比学习用特征
        pre_contrast = torch.cat([pre_dce_attn, pre_dwi_attn], dim=1)  # [B,1024]
        post_contrast = torch.cat([post_dce_contrast, post_dwi_contrast], dim=1)  # [B,1024]
        # 单模态特征（用于delta损失）
        pre_dce = pre_dce_base
        pre_dwi =  pre_dwi_base
        post_dce =  post_dce_base
        post_dwi = post_dwi_base
        return pre_pcr, post_pcr, pre_contrast, post_contrast, pre_dce, pre_dwi, post_dce, post_dwi


class FullModel(nn.Module):
    """完整的Encoder-Decoder架构"""
    def __init__(self, device):
        super().__init__()
        self.encoder = HybridEncoder(device)
        # 新增pCR分类器
        self.pcr_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        # 可学习的损失权重
        self.weights = nn.ParameterDict({
            'class': nn.Parameter(torch.tensor(1.5)),
            'contrast': nn.Parameter(torch.tensor(1.0)),
            'kl': nn.Parameter(torch.tensor(0.5)),
            'delta': nn.Parameter(torch.tensor(1.0)),
            'proto': nn.Parameter(torch.tensor(0.5))
        })
        # self.weights = nn.ParameterDict({
        #     'class': nn.Parameter(torch.tensor(2.0)),
        #     'contrast': nn.Parameter(torch.tensor(1.0)),
        #     'delta': nn.Parameter(torch.tensor(1.0))
        # })
        self.margin = nn.Parameter(torch.tensor(1.0))
        # 新增类别平衡参数
        self.class_weights = nn.Parameter(
            torch.tensor([1.0, 5.0]),  # [weight_for_class0, weight_for_class1]
            requires_grad=False  # 可设为True让模型自动学习
        )
        self.device = device
        self.to(device)

    def contrastive_learning(self, pre_feat, post_feat, pcr_label):
        distance = F.mse_loss(pre_feat, post_feat, reduction='none').mean(dim=1)
        loss_pcr1 = F.relu(self.margin - distance)  # pCR=1的损失
        loss_pcr0 = F.relu(distance - self.margin)  # pCR=0的损失
        
        # 按标签选择对应损失
        loss = torch.where(pcr_label == 1, loss_pcr1, loss_pcr0)
        return loss.mean()

    def kl_divergence_loss(self, pre_feature, post_feature):
        temperature = 0.1
        pre_dist = F.log_softmax(pre_feature/temperature, dim=1)
        post_dist = F.softmax(post_feature/temperature, dim=1).clamp(min=1e-6)
        return F.kl_div(pre_dist, post_dist, reduction='batchmean')

    def delta_feat_loss(self, pre_feature, post_feature):
        cosine_sim = F.cosine_similarity(pre_feature, post_feature, dim=1)
        return (1 - cosine_sim).mean()
        # delta_feat = torch.abs(pre_feature - post_feature).mean(dim=1)
        # return ((1 - pcr_label) * delta_feat).mean()

    def prototype_loss(self, pre_feature, post_feature, pcr_label):
        """
        计算 Prototype Loss，用于度量样本与类别原型的距离
        :param pre_feature: (batch_size, feature_dim) 治疗前的特征
        :param post_feature: (batch_size, feature_dim) 治疗后的特征
        :param pcr_label: (batch_size,) 样本的 pCR 标签，0 或 1
        :return: Prototype Loss
        """

        # 计算 pCR=0 的原型（如果没有样本，则使用全零向量）
        if (pcr_label == 0).sum() > 0:
            proto_0_pre = pre_feature[pcr_label == 0].mean(dim=0, keepdim=True)
            proto_0_post = post_feature[pcr_label == 0].mean(dim=0, keepdim=True)
        else:
            proto_0_pre = torch.zeros((1, pre_feature.size(1)), device=pre_feature.device)
            proto_0_post = torch.zeros((1, post_feature.size(1)), device=post_feature.device)

        # 计算 pCR=1 的原型（如果没有样本，则使用全零向量）
        if (pcr_label == 1).sum() > 0:
            proto_1_pre = pre_feature[pcr_label == 1].mean(dim=0, keepdim=True)
            proto_1_post = post_feature[pcr_label == 1].mean(dim=0, keepdim=True)
        else:
            proto_1_pre = torch.zeros((1, pre_feature.size(1)), device=pre_feature.device)
            proto_1_post = torch.zeros((1, post_feature.size(1)), device=post_feature.device)

        # 计算治疗前特征到对应原型的 MSE 损失
        dist_0_pre = F.mse_loss(pre_feature, proto_0_pre.expand_as(pre_feature), reduction='none').mean(dim=1)
        dist_1_pre = F.mse_loss(pre_feature, proto_1_pre.expand_as(pre_feature), reduction='none').mean(dim=1)

        # 计算治疗后特征到对应原型的 MSE 损失
        dist_0_post = F.mse_loss(post_feature, proto_0_post.expand_as(post_feature), reduction='none').mean(dim=1)
        dist_1_post = F.mse_loss(post_feature, proto_1_post.expand_as(post_feature), reduction='none').mean(dim=1)

        # 计算最终的 Prototype Loss
        pcr_label = pcr_label.float()  # 确保计算时数据类型正确
        proto_loss = (1 - pcr_label) * (dist_0_pre + dist_0_post) + pcr_label * (dist_1_pre + dist_1_post)

        return proto_loss.mean()

    def forward(self, pre_dce, pre_dwi, post_dce, post_dwi, pcr_label):
        # 编码阶段
        pre_pcr, post_pcr, pre_contrast, post_contrast, pre_dce, pre_dwi, post_dce, post_dwi = self.encoder(pre_dce, pre_dwi, post_dce, post_dwi)
        pcr_pred = self.pcr_classifier(pre_pcr)

        # 对比学习
        contrast_loss = self.contrastive_learning(pre_contrast, post_contrast, pcr_label)

        kl_loss = self.kl_divergence_loss(pre_contrast, post_contrast)

        delta_loss1 = self.delta_feat_loss(pre_dce, post_dce)
        delta_loss2 = self.delta_feat_loss(pre_dwi, post_dwi)

        proto_loss = self.prototype_loss(pre_pcr, post_pcr, pcr_label)

        # 改进的加权分类损失
        # weight = self.class_weights[pcr_label.long()]
        # cls_loss = F.binary_cross_entropy(
        #     pcr_pred, 
        #     pcr_label.unsqueeze(1).float(),
        #     weight=weight.unsqueeze(1)
        # )
        with torch.no_grad():
            pos_ratio = pcr_label.float().mean()
            class_weight = torch.clamp(1.0 / (pos_ratio + 1e-6), min=1.0, max=10.0)

        pcr_label = pcr_label.float()
        cls_loss = F.binary_cross_entropy(
            pcr_pred, 
            pcr_label.unsqueeze(1),
            weight=class_weight*pcr_label.unsqueeze(1)+(1-pcr_label).unsqueeze(1)
        )
        
        # # 动态调整对比损失权重（基于类别比例）
        # with torch.no_grad():
        #     pcr_ratio = pcr_label.float().mean()
        #     contrast_weight = 1.0 + (0.5 - pcr_ratio)  # 当pCR=1样本少时增加权重

        # total_loss = self.weights['class'] * cls_loss + self.weights['contrast'] * contrast_loss + self.weights['delta'] * (delta_loss1 + delta_loss2)
        total_loss = cls_loss + self.weights['contrast'] * contrast_loss + self.weights['kl'] * kl_loss + self.weights['delta'] * (delta_loss1 + delta_loss2) + self.weights['proto'] * proto_loss
        return cls_loss, total_loss



# # 检查 GPU 是否可用
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 实例化模型
# model = FullModel(device)

# # 准备测试数据
# batch_size = 32
# pre_dce = torch.randn(batch_size, 1, 32, 96, 96).to(device)
# pre_dwi = torch.randn(batch_size, 1, 32, 96, 96).to(device)
# post_dce = torch.randn(batch_size, 1, 32, 96, 96).to(device)
# post_dwi = torch.randn(batch_size, 1, 32, 96, 96).to(device)
# pcr_label = torch.randint(0, 2, (batch_size,)).to(device)

# # 执行前向传播
# with torch.no_grad():
#     cls_loss, total_loss = model(pre_dce, pre_dwi, post_dce, post_dwi, pcr_label)

# # 打印结果
# print("class loss:", cls_loss.item())
# print("Total loss:", total_loss.item())



# 第二阶段微调预测
class AsymCrossAttention(nn.Module):
    def __init__(self, img_dim=256, clinical_dim=32):
        super().__init__()
        # 临床特征升维路径 (32 -> 128 -> 256)
        self.clinical_up = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, img_dim),
            nn.LayerNorm(img_dim)
        )
        
        # 图像特征降维路径 (256 -> 128)
        self.img_down = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128)
        )
        
        # 注意力机制参数
        self.query = nn.Linear(128, 64, bias=False)  # Q来自降维后的图像
        self.key = nn.Linear(img_dim, 64, bias=False) # K来自升维后的临床，调整为64
        self.value = nn.Linear(img_dim, 128)           # V来自升维后的临床，调整为128
        
        # 残差连接前的升维 (128 -> 256)
        self.res_up = nn.Sequential(
            nn.Linear(128, img_dim),
            nn.LayerNorm(img_dim)
        )
        
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for m in [self.query, self.key, self.value]:
            nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, img_feat, clinical_feat):
        """
        输入:
            img_feat: [B, 256]     图像特征
            clinical_feat: [B, 32] 临床特征
        输出:
            [B, 256] 增强后的图像特征
        """
        # 1. 模态对齐
        clinical_high = self.clinical_up(clinical_feat)  # [B, 32] -> [B, 256]
        img_low = self.img_down(img_feat)               # [B, 256] -> [B, 128]
        
        # 2. 注意力计算
        Q = self.query(img_low)  # [B, 64]
        K = self.key(clinical_high)  # [B, 64]
        V = self.value(clinical_high)  # [B, 128]
        
        # 3. 注意力权重
        attn_weights = torch.softmax((Q @ K.transpose(0, 1)) / (64**0.5), dim=-1)  # [B, B]
        attended = (attn_weights @ V)  # [B, 128]
        
        # 4. 残差连接
        return img_feat + self.res_up(attended)  # [B, 256]


# 替换AsymCrossAttention为更稳健的融合
class FeatureFusion(nn.Module):
    def __init__(self, img_dim=256, clinical_dim=32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(img_dim + clinical_dim, img_dim),
            nn.Sigmoid()
        )
        
    def forward(self, img_feat, clinical_feat):
        gate_value = self.gate(torch.cat([img_feat, clinical_feat], dim=1))
        print(f"img_feat shape: {img_feat.shape}")
        print(f"clinical_feat shape: {clinical_feat.shape}")
        print(f"gate_value shape: {gate_value.shape}")
        return img_feat * gate_value + clinical_feat * (1 - gate_value)


class FineTuningModel(nn.Module):
    def __init__(self, encoder, clinical_features, experiment_type, device):
        super(FineTuningModel, self).__init__()
        self.encoder = encoder

        # 临床数据分支（使用更复杂的 MLP 结构）
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
        # ========== 改进点 1：影像特征 Adapter ==========
        self.image_adapter = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # self.fusion = AsymCrossAttention(img_dim=256, clinical_dim=32)
        self.fusion = FeatureFusion(img_dim=256, clinical_dim=32)

        if experiment_type == 'clinical_only':
            self.classifier = nn.Linear(32, 1) 
        else:
            self.classifier = nn.Sequential(
                nn.Linear(256+32, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        self.device = device
        self.to(device)

    def forward_clinical_only(self, clinical_data):
        clinical_features = self.clinical_net(clinical_data)
        pcr_pred = self.classifier(clinical_features)
        return pcr_pred

    def forward(self, pre_dce, pre_dwi, clinical_data):
        # 使用 encoder 提取治疗前 DCE 和 DWI 的特征
        pre_x_dce = self.encoder.encoder_c(pre_dce)
        pre_x_dwi = self.encoder.encoder_dwi(pre_dwi)

        pre_dce_pcr = self.encoder.pcr_proj(pre_x_dce)
        pre_dwi_pcr = self.encoder.pcr_proj(pre_x_dwi)

        # ====== 改进点 1：影像特征适配器 ======
        pre_dce_feat = self.image_adapter(pre_dce_pcr)
        pre_dwi_feat = self.image_adapter(pre_dwi_pcr)
        pre_fused_feat = torch.cat((pre_dce_feat, pre_dwi_feat), dim=1)  # [B, 256]

        clinical_features = self.clinical_net(clinical_data)
        # 影像 + 临床融合（自适应加权）
        # fused_feat = self.fusion(pre_fused_feat, clinical_features)
        fused_feat = torch.cat((pre_fused_feat, clinical_features), dim=1)

        pcr_pred = self.classifier(fused_feat)
        # pcr_pred = torch.sigmoid(pcr_pred)
        return fused_feat, pcr_pred

    # 替换原有loss函数为更平衡的版本
    def loss(self, pre_fused_feat, pcr_pred, pcr_label):

        # 使用带类别权重的BCE
        weight = torch.where(pcr_label == 1, 
                        torch.tensor(5.0, device=self.device),
                        torch.tensor(1.0, device=self.device))

        # 1. 改进的Focal Loss
        bce_loss = F.binary_cross_entropy(
            pcr_pred.squeeze(1), 
            pcr_label.float(),
            weight=weight
        )
        # pt = torch.exp(-bce_loss)
        # focal_loss = (0.25 * (1-pt)**2 * bce_loss).mean()
        
        # 2. 添加Dice Loss增强recall
        pred_sigmoid = torch.sigmoid(pcr_pred.squeeze(1))
        intersection = (pred_sigmoid * pcr_label).sum()
        dice_loss = 1 - (2. * intersection + 1) / (pred_sigmoid.sum() + pcr_label.sum() + 1)
        
        # 3. 平衡的正则化项
        reg_loss = 0.01 * (pre_fused_feat.norm(p=2, dim=1).mean())
        
        total_loss = bce_loss + dice_loss + reg_loss
        return total_loss, bce_loss




# class FineTuningModel(nn.Module):
#     def __init__(self, encoder, clinical_features, experiment_type, device):
#         super(FineTuningModel, self).__init__()
#         self.encoder = encoder

#         # 临床数据分支（使用更复杂的 MLP 结构）
#         self.clinical_net = nn.Sequential(
#             nn.Linear(clinical_features, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
            
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
            
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU()
#         )
#         # 新增类别平衡措施
#         self.focal_loss_alpha = 0.75  # 调节因子
#         self.focal_loss_gamma = 2.0   # 难样本聚焦参数

#         if experiment_type == 'clinical_only':
#             self.classifier = nn.Linear(32, 1) 
#         else:
#             self.classifier = nn.Linear(512 + 32, 1) 

#         self.device = device
#         self.to(device)

#     def forward_clinical_only(self, clinical_data):
#         clinical_features = self.clinical_net(clinical_data)
#         pcr_pred = self.classifier(clinical_features)
#         return pcr_pred

#     def forward(self, pre_dce, pre_dwi, clinical_data):
#         # 使用 encoder 提取治疗前 DCE 和 DWI 的特征
#         pre_x_dce = self.encoder.encoder_c(pre_dce)
#         pre_x_dwi = self.encoder.encoder_dwi(pre_dwi)

#         pre_dce_pcr = self.encoder.pcr_proj(pre_x_dce)
#         pre_dwi_pcr = self.encoder.pcr_proj(pre_x_dwi)

#         # 融合治疗前的影像特征
#         pre_fused_feat = torch.cat((pre_dce_pcr, pre_dwi_pcr), dim=1)

#         clinical_features = self.clinical_net(clinical_data)
#         # 影像 + 临床融合（自适应加权）
#         fused_feat = torch.cat((pre_fused_feat, clinical_features), dim=1)

#         pcr_pred = self.classifier(fused_feat)
#         pcr_pred = torch.sigmoid(pcr_pred)
#         return fused_feat, pcr_pred


#     # def loss(self, pre_fused_feat, pcr_pred, pcr_label):
#     #     """最终优化版 Triplet Loss：稳定性 + Hard Mining + 随机性平衡"""
        
#     #     # 1️⃣ 分类损失
#     #     class_loss = F.binary_cross_entropy(pcr_pred, pcr_label.unsqueeze(1))
        
#     #     # 2️⃣ 获取正负样本索引
#     #     pos_idx = (pcr_label == 1).nonzero(as_tuple=True)[0]
#     #     neg_idx = (pcr_label == 0).nonzero(as_tuple=True)[0]
#     #     triplet_loss_val = torch.tensor(0.0, device=self.device)  # 初始化

#     #     # 3️⃣ 确保至少有 1 个正样本和 1 个负样本
#     #     if len(pos_idx) > 0 and len(neg_idx) > 0:
#     #         # 选择 Anchor（从正样本中随机选择）
#     #         anchor_idx = pos_idx[torch.randint(0, len(pos_idx), (1,))]
#     #         anchor = pre_fused_feat[anchor_idx]

#     #         # 选择 Positive（确保不等于 anchor）
#     #         remaining_pos = pos_idx[pos_idx != anchor_idx] if len(pos_idx) > 1 else pos_idx
#     #         positive_idx = remaining_pos[torch.randint(0, len(remaining_pos), (1,))]
#     #         positive = pre_fused_feat[positive_idx]

#     #         # 选择 Negative（Hard Mining + 随机性平衡）
#     #         neg_samples = pre_fused_feat[neg_idx]
#     #         dists = torch.norm(neg_samples - anchor, dim=1)
#     #         hard_neg = neg_samples[torch.argmin(dists)]
#     #         rand_neg = neg_samples[torch.randint(0, len(neg_idx), (1,))]
#     #         negative = 0.7 * hard_neg + 0.3 * rand_neg  # 70% Hard Mining, 30% 随机

#     #         # 计算 Triplet Loss
#     #         triplet_loss_val = F.triplet_margin_loss(
#     #             anchor=anchor.reshape(1, -1),
#     #             positive=positive.reshape(1, -1),
#     #             negative=negative.reshape(1, -1),
#     #             margin=1.0,
#     #             reduction='mean'
#     #         )

#     #     # 4️⃣ 组合损失
#     #     total_loss = class_loss + 0.3 * triplet_loss_val
#     #     return total_loss, class_loss


#     def loss(self, pre_fused_feat, pcr_pred, pcr_label):
#             """改进的混合损失函数"""
#             # 1. Focal Loss解决类别不平衡
#             bce_loss = F.binary_cross_entropy(
#                 pcr_pred, 
#                 pcr_label.unsqueeze(1).float(),
#                 reduction='none'
#             )
#             pt = torch.exp(-bce_loss)
#             focal_loss = (self.focal_loss_alpha * (1-pt)**self.focal_loss_gamma * bce_loss).mean()
            
#             # 2. 改进的Triplet Loss（动态margin）
#             pos_idx = (pcr_label == 1).nonzero(as_tuple=True)[0]
#             neg_idx = (pcr_label == 0).nonzero(as_tuple=True)[0]
#             triplet_loss = torch.tensor(0.0, device=self.device)
            
#             if len(pos_idx) > 1 and len(neg_idx) > 0:  # 至少需要2个正样本
#                 # 动态margin（基于类别比例）
#                 margin = 1.0 + (1.0 - pcr_label.float().mean())  # pCR=1样本少时增大margin
                
#                 # 在线困难样本挖掘
#                 anchors = pre_fused_feat[pos_idx]
#                 positives = pre_fused_feat[pos_idx.roll(1)]  # 循环取正样本
#                 negatives = pre_fused_feat[neg_idx]
                
#                 # 计算所有可能的triplet
#                 pos_dists = torch.cdist(anchors, positives)
#                 neg_dists = torch.cdist(anchors, negatives)
                
#                 # 选择最困难的负样本
#                 hardest_neg_dists, _ = neg_dists.min(dim=1)
#                 triplet_loss = F.relu(pos_dists.diag() - hardest_neg_dists + margin).mean()
            
#             # 3. 原型对比损失（增强类内紧凑性）
#             proto_loss = self.prototype_contrast_loss(pre_fused_feat, pcr_label)
            
#             total_loss = focal_loss + 0.2 * triplet_loss + 0.1 * proto_loss
#             return total_loss, focal_loss
        
#     def prototype_contrast_loss(self, features, labels):
#         """新增的原型对比损失"""
#         with torch.no_grad():
#             # 计算类原型
#             proto_0 = features[labels == 0].mean(dim=0) if (labels == 0).any() else features[0]
#             proto_1 = features[labels == 1].mean(dim=0) if (labels == 1).any() else features[0]
        
#         # 计算样本到原型的距离
#         dist_to_0 = torch.norm(features - proto_0, dim=1)
#         dist_to_1 = torch.norm(features - proto_1, dim=1)
        
#         # 鼓励样本靠近所属类原型
#         loss = torch.where(
#             labels == 1,
#             dist_to_1 - dist_to_0 + 0.5,  # pCR=1样本应更接近proto_1
#             dist_to_0 - dist_to_1 + 0.5    # pCR=0样本应更接近proto_0
#         )
#         return F.relu(loss).mean()




# def loss(self, pre_fused_feat, pcr_pred, pcr_label):
#             """改进的混合损失函数"""
#             # 1. Focal Loss解决类别不平衡
#             bce_loss = F.binary_cross_entropy_with_logits(
#                 pcr_pred.squeeze(1), 
#                 pcr_label.float(),
#                 reduction='none'
#             )
#             pt = torch.exp(-bce_loss)
#             focal_loss = (self.focal_loss_alpha * (1-pt)**self.focal_loss_gamma * bce_loss).mean()
            
#             # 2. 改进的Triplet Loss（动态margin）
#             pos_idx = (pcr_label == 1).nonzero(as_tuple=True)[0]
#             neg_idx = (pcr_label == 0).nonzero(as_tuple=True)[0]
#             triplet_loss = torch.tensor(0.0, device=self.device)
            
#             if len(pos_idx) > 1 and len(neg_idx) > 0:  # 至少需要2个正样本
#                 # 动态margin（基于类别比例）
#                 margin = 1.0 + (1.0 - pcr_label.float().mean())  # pCR=1样本少时增大margin
                
#                 # 在线困难样本挖掘
#                 anchors = pre_fused_feat[pos_idx]
#                 positives = pre_fused_feat[pos_idx.roll(1)]  # 循环取正样本
#                 negatives = pre_fused_feat[neg_idx]
                
#                 # 计算所有可能的triplet
#                 pos_dists = torch.cdist(anchors, positives)
#                 neg_dists = torch.cdist(anchors, negatives)
                
#                 # 选择最困难的负样本
#                 hardest_neg_dists, _ = neg_dists.min(dim=1)
#                 triplet_loss = F.relu(pos_dists.diag() - hardest_neg_dists + margin).mean()
            
#             # 3. 原型对比损失（增强类内紧凑性）
#             proto_loss = self.prototype_contrast_loss(pre_fused_feat, pcr_label)
            
#             total_loss = focal_loss + 0.2 * triplet_loss + 0.1 * proto_loss
#             return total_loss, focal_loss
        
# def prototype_contrast_loss(self, features, labels):
#     """新增的原型对比损失"""
#     with torch.no_grad():
#         # 计算类原型
#         proto_0 = features[labels == 0].mean(dim=0) if (labels == 0).any() else features[0]
#         proto_1 = features[labels == 1].mean(dim=0) if (labels == 1).any() else features[0]
    
#     # 计算样本到原型的距离
#     dist_to_0 = torch.norm(features - proto_0, dim=1)
#     dist_to_1 = torch.norm(features - proto_1, dim=1)
    
#     # 鼓励样本靠近所属类原型
#     loss = torch.where(
#         labels == 1,
#         dist_to_1 - dist_to_0 + 0.5,  # pCR=1样本应更接近proto_1
#         dist_to_0 - dist_to_1 + 0.5    # pCR=0样本应更接近proto_0
#     )
#     return F.relu(loss).mean()




# # 定义设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 初始化 encoder
# encoder = HybridEncoder(device)

# # 定义临床特征数量
# clinical_features = 10

# # 初始化 FineTuningModel
# model = FineTuningModel(encoder, clinical_features, experiment_type='both', device=device)

# # 模拟输入数据
# batch_size = 32
# pre_dce = torch.randn(batch_size, 1, 32, 96, 96).to(device)
# pre_dwi = torch.randn(batch_size, 1, 32, 96, 96).to(device)
# clinical_data = torch.randn(batch_size, clinical_features).to(device)
# pcr_label = torch.randint(0, 2, (batch_size,)).float().to(device)

# # 前向传播
# pre_fused_feat, pcr_pred = model(pre_dce, pre_dwi, clinical_data)

# # 计算损失
# total_loss, focal_loss = model.loss(pre_fused_feat, pcr_pred, pcr_label)

# print(f"预测结果: {pcr_pred}")
# print(f"总损失: {total_loss.item()}")


