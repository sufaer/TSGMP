import torch
import torch.nn as nn
import torch.nn.functional as F


# 全局损失
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2):
        super().__init__()
        # 确保alpha是正确形状的Tensor
        if alpha is None:
            self.register_buffer('alpha', torch.ones(class_num))
        else:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

        self.gamma = gamma
        self.class_num = class_num

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha = self.alpha[targets.long()]
        loss = alpha * (1-pt)**self.gamma * ce_loss
        return loss.mean()
  

# 跨模态融合
class CMFA(nn.Module):
    def __init__(self, img_dim, tab_dim, hid_dim, heads=4, dropout=0.2):
        super().__init__()

        self.fi1 = nn.Linear(img_dim, hid_dim)
        self.fi2 = nn.Linear(hid_dim, hid_dim)
        self.ft1 = nn.Linear(tab_dim, hid_dim)
        self.ft2 = nn.Linear(hid_dim, hid_dim)

        self.conv_i1 = nn.Linear(hid_dim, hid_dim)
        self.conv_i2 = nn.Linear(hid_dim, hid_dim)
        self.conv_i3 = nn.Linear(hid_dim, hid_dim)
        self.conv_t1 = nn.Linear(hid_dim, hid_dim)
        self.conv_t2 = nn.Linear(hid_dim, hid_dim)
        self.conv_t3 = nn.Linear(hid_dim, hid_dim)

        self.self_attn_V = nn.MultiheadAttention(hid_dim, heads, dropout=dropout)
        self.self_attn_T = nn.MultiheadAttention(hid_dim, heads, dropout=dropout)
        
    # def forward(self, i, t):
    #     i_ = F.relu(self.fi1(i))
    #     t_ = F.relu(self.ft1(t))
    #     residual_i_ = i_
    #     residual_t_ = t_

    #     v1 = F.relu(self.conv_i1(i_))
    #     k1 = F.relu(self.conv_i2(i_))
    #     q1 = F.relu(self.conv_i3(i_))
    #     v2 = F.relu(self.conv_t1(t_))
    #     k2 = F.relu(self.conv_t2(t_))
    #     q2 = F.relu(self.conv_t3(t_))

    #     V_ = self.self_attn_V(q2, k1, v1)[0]
    #     T_ = self.self_attn_T(q1, k2, v2)[0]
    #     V_ = V_ + residual_i_
    #     T_ = T_ + residual_t_

    #     V_ = self.fi2(V_)
    #     T_ = self.ft2(T_)

    #     return torch.cat((V_, T_), dim=1)
    
    def forward(self, i, t):
        # i: token_img, 形状为 [B, img_dim]
        # t: token_rad_clin, 形状为 [B, tab_dim]

        # 将 2 维张量通过线性层转换为 hid_dim 维度的 2 维张量
        i_ = F.relu(self.fi1(i)) # [B, hid_dim]
        t_ = F.relu(self.ft1(t)) # [B, hid_dim]
        residual_i_ = i_
        residual_t_ = t_

        # 为输入张量添加一个序列长度维度，形状变为 [B, 1, hid_dim]
        i_unsqueezed = i_.unsqueeze(0) # 将批次维度放在第一位，与 MultiheadAttention 的 batch_first 参数有关
        t_unsqueezed = t_.unsqueeze(0)

        # 通过线性层生成 q, k, v，维度仍为 [B, 1, hid_dim]
        v1 = F.relu(self.conv_i1(i_unsqueezed))
        k1 = F.relu(self.conv_i2(i_unsqueezed))
        q1 = F.relu(self.conv_i3(i_unsqueezed))
        v2 = F.relu(self.conv_t1(t_unsqueezed))
        k2 = F.relu(self.conv_t2(t_unsqueezed))
        q2 = F.relu(self.conv_t3(t_unsqueezed))

        # 调用 MultiheadAttention
        V_ = self.self_attn_V(q2, k1, v1)[0]
        T_ = self.self_attn_T(q1, k2, v2)[0]
        
        # 移除序列长度维度，将张量变回 2 维
        V_ = V_.squeeze(0) # [B, hid_dim]
        T_ = T_.squeeze(0) # [B, hid_dim]

        # ... (后续操作保持不变)
        V_ = V_ + residual_i_
        T_ = T_ + residual_t_

        V_ = self.fi2(V_)
        T_ = self.ft2(T_)

        return torch.cat((V_, T_), dim=1)


# 临床数据处理模块 —— MLP
class ClinicalProcessor(nn.Module):
    def __init__(self, input_dim=8, d_model=256, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: [B, clinical_dim]
        Returns:
            [B, d_model]
        """
        return self.net(x)


# Rad数据处理模块 —— MLP
class RadiomicsProcessor(nn.Module):
    def __init__(self, input_dim=2264, d_model=256, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: [B, clinical_dim]
        Returns:
            [B, d_model]
        """
        return self.net(x)


# 跨模态交互
class CrossModalAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.head_dim = d_model // n_heads  # 显式定义每个头的维度
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
    def forward(self, clin_feat, rad_feat):
        Q = self.query(clin_feat)
        K = self.key(rad_feat)
        V = self.value(rad_feat)
        attn = F.softmax(Q @ K.T / torch.sqrt(torch.tensor(self.head_dim, device=Q.device)), dim=-1)
        return attn @ V
    

# 动态特征加权
class DynamicFusion(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model*2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, clin_feat, rad_feat):
        weights = self.gate(torch.cat([clin_feat, rad_feat], dim=-1))
        return weights[:, 0:1] * clin_feat + weights[:, 1:2] * rad_feat


# 主融合模型
class DoubleTower(nn.Module):
    def __init__(self, in_channel=1, clinical_dim=23, rad_dim=2264, d_model=256, dropout_rate=0.25, num_classes=2, weights_path=None, focal_alpha=None, focal_gamma=2.0, device=torch.device("cuda")):
        super().__init__()
        self.device = device
        # 初始化各处理器
        # self.image_processor = MedicalImageProcessor_CMT(in_channel=in_channel,d_model=d_model, dropout=dropout_rate, weights_path=weights_path, device=device)
        # self.image_processor = MedicalImageProcessor_Spectral(in_channel=in_channel,d_model=d_model, dropout=dropout_rate,weights_path=weights_path, device=device)
        self.rad_processor = RadiomicsProcessor(input_dim=rad_dim, d_model=d_model, dropout=dropout_rate)
        self.clin_processor = ClinicalProcessor(input_dim=clinical_dim, d_model=d_model, dropout=dropout_rate)

        self.cross_attn = CrossModalAttention(d_model=d_model)
        self.cmfa = CMFA(img_dim=d_model, tab_dim=d_model, hid_dim=d_model, heads=4, dropout=dropout_rate)
        self.fusion = DynamicFusion(d_model=d_model)
       
        # 分类器，增加非线性层数
        self.classifier = nn.Sequential(
            nn.Linear(d_model*2, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        ).to(device)

         # === 新增损失函数配置 ===
        self.focal_loss = FocalLoss(
            class_num=num_classes,
            alpha=focal_alpha,  # 建议设置为[1-pCR_rate, pCR_rate]
            gamma=focal_gamma
        )


    def forward(self,  x1, x2, x_clin, x_rad, label=None):
        # 处理各模态数据
        # 只用临床特征进行拼接得到的预测结果
        token_img, _, _ = self.image_processor(x1, x2) # [B,256]
        token_rad = self.rad_processor(x_rad) # [B,256]
        token_clin = self.clin_processor(x_clin) # [B,256]

        # 增强的交互模块 —— 分类器的输入维度是 256
        attn_enhanced = self.cross_attn(token_clin, token_rad)
        token_rad_clin = self.fusion(token_clin, attn_enhanced) # [B,256]

        tokens = self.cmfa(token_img, token_rad_clin) # [B,512]
        # tokens = torch.concat([token_img, token_rad_clin], dim=-1)
        # tokens = self.cross_attn(token_rad_clin,token_img)

        logits = self.classifier(tokens)
        
        # === 多任务损失计算 ===
        if label is not None:
            # 1. Focal Loss
            fl = self.focal_loss(logits, label)
            
            # 动态加权总损失
            total_loss = fl
            
            return logits, total_loss
        return logits    

