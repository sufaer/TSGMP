import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# --- 1. 核心模块（2D版本）---

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super(Conv2dLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):
        return self.conv2d(x)

class ResBlock2d(nn.Module):
    def __init__(self, channels):
        super(ResBlock2d, self).__init__()
        self.conv1 = Conv2dLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.gelu = GELU()
        self.conv2 = Conv2dLayer(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out = self.conv2(out)
        out = residual + out
        return self.gelu(out)

class ChannelAttention2d(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            Conv2dLayer(in_channels, in_channels // reduction_ratio, 1, 1, 0),
            GELU(),
            Conv2dLayer(in_channels // reduction_ratio, in_channels, 1, 1, 0)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention2d(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention2d, self).__init__()
        padding = kernel_size // 2
        self.conv2d = Conv2dLayer(2, 1, kernel_size, 1, padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_stack = torch.cat([avg_out, max_out], dim=1)
        out = self.conv2d(x_stack)
        return self.sigmoid(out)

class CBAM2d(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM2d, self).__init__()
        self.channel_attention = ChannelAttention2d(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention2d(kernel_size)

    def forward(self, x):
        channel_attn_weight = self.channel_attention(x)
        x = x * channel_attn_weight
        spatial_attn_weight = self.spatial_attention(x)
        x = x * spatial_attn_weight
        return x

# --- 2. 特征提取与重构模块（2D版本）---

class ResNetEncoder2d(nn.Module):
    def __init__(self, in_channels, base_channels=32, num_down_blocks=3):
        super(ResNetEncoder2d, self).__init__()
        self.initial_conv = nn.Sequential(
            Conv2dLayer(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            GELU()
        )
        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        for i in range(num_down_blocks):
            self.down_blocks.append(
                nn.Sequential(
                    ResBlock2d(current_channels),
                    nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, stride=2, padding=1),
                    GELU()
                )
            )
            current_channels *= 2
        
        self.bottleneck = ResBlock2d(current_channels)

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.bottleneck(x)
        return x

class Decoder2d(nn.Module):
    def __init__(self, out_channels, base_channels=32, num_up_blocks=3, encoder_bottleneck_channels=None):
        super(Decoder2d, self).__init__()
        self.up_blocks = nn.ModuleList()
        current_channels = encoder_bottleneck_channels
        
        for i in range(num_up_blocks):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_channels, current_channels // 2, kernel_size=4, stride=2, padding=1),
                    ResBlock2d(current_channels // 2) 
                )
            )
            current_channels //= 2
        
        self.final_conv = nn.Sequential(
            Conv2dLayer(base_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        for block in self.up_blocks:
            x = block(x)
        x = self.final_conv(x)
        return x

# --- 3. 融合模块（2D版本）---

class FusionModule_v2(nn.Module):
    def __init__(self, in_channels):
        super(FusionModule_v2, self).__init__()
        # in_channels 现在是拼接后的通道数，例如 encoder_bottleneck_channels * 3
        self.cbam = CBAM2d(in_channels)

    def forward(self, fused_features): # <-- 只需要一个参数
        return self.cbam(fused_features)


# 临床对比损失
class ContrastiveLoss_euc(nn.Module):  #计算欧氏距离
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):  #具有差异性，label为1；不具有差异性，label为0  label为1时，希望欧氏距离更小
        output1 = torch.mean(output1, dim=1)
        output2 = torch.mean(output2, dim=1)
        euclidean_distance = F.pairwise_distance(output1, output2).reshape(-1,1)
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


# --- 4. 整体模型（2D版本）---

class DoubleTower_Delta(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, num_down_blocks=3, contrast_margin=2):
        super(DoubleTower_Delta, self).__init__()
        
        # 编码器。现在我们只需要每个模态一个编码器。
        self.encoder_dce = ResNetEncoder2d(in_channels, base_channels, num_down_blocks)
        self.encoder_dwi = ResNetEncoder2d(in_channels, base_channels, num_down_blocks)
        
        encoder_bottleneck_channels = base_channels * (2 ** num_down_blocks)
        
        # 解码器，用于重构任务
        self.decoder_dce = Decoder2d(out_channels, base_channels, num_down_blocks, encoder_bottleneck_channels)
        self.decoder_dwi = Decoder2d(out_channels, base_channels, num_down_blocks, encoder_bottleneck_channels)
        
        # 融合模块
        # 由于我们现在要融合 pre, post, delta 三种特征，特征通道数变为 3 倍
        self.fusion_module_dce = FusionModule_v2(encoder_bottleneck_channels * 3)
        self.fusion_module_dwi = FusionModule_v2(encoder_bottleneck_channels * 3)

        self.reconstruction_loss = nn.MSELoss()
        self.contrastive_loss = ContrastiveLoss_euc(margin=contrast_margin)
        
        # 辅助分类器
        # 输入通道数是两个模态的融合特征（每个模态是3倍通道数）
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_bottleneck_channels * 3 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, pre_dce, post_dce, pre_dwi, post_dwi, pcr_label):
        
        # --- 1. 编码器阶段 ---
        # 编码 pre 和 post 图像
        encoded_pre_dce = self.encoder_dce(pre_dce)
        encoded_post_dce = self.encoder_dce(post_dce)
        encoded_pre_dwi = self.encoder_dwi(pre_dwi)
        encoded_post_dwi = self.encoder_dwi(post_dwi)

        # --- 2. 在特征层面计算 delta ---
        # 确保 pre 和 post 图像经过了相同的增强操作，其特征图是空间对齐的
        delta_features_dce = encoded_post_dce - encoded_pre_dce
        delta_features_dwi = encoded_post_dwi - encoded_pre_dwi
        
        # --- 3. 重构损失阶段 ---
        # 重构损失仍然使用 encoded_pre 和 encoded_post
        reconstructed_pre_dce = self.decoder_dce(encoded_pre_dce) 
        reconstructed_post_dce = self.decoder_dce(encoded_post_dce)
        reconstructed_pre_dwi = self.decoder_dwi(encoded_pre_dwi)
        reconstructed_post_dwi = self.decoder_dwi(encoded_post_dwi)

        reco_loss_pre_dce = self.reconstruction_loss(reconstructed_pre_dce, pre_dce)
        reco_loss_post_dce = self.reconstruction_loss(reconstructed_post_dce, post_dce)
        reco_loss_pre_dwi = self.reconstruction_loss(reconstructed_pre_dwi, pre_dwi)
        reco_loss_post_dwi = self.reconstruction_loss(reconstructed_post_dwi, post_dwi)
        
        total_reco_loss = reco_loss_pre_dce + reco_loss_post_dce + reco_loss_pre_dwi + reco_loss_post_dwi

        # --- 4. 对比损失阶段 ---
        contrast_label = 1 - pcr_label.float()
        loss_dce = self.contrastive_loss(encoded_pre_dce, encoded_post_dce, contrast_label)
        loss_dwi = self.contrastive_loss(encoded_pre_dwi, encoded_post_dwi, contrast_label)
        contrast_loss = loss_dce + loss_dwi
        
        # --- 5. 辅助任务：直接对融合特征进行分类 ---
        # 将 pre, post, delta_features 三种特征拼接
        fused_features_dce = torch.cat([encoded_pre_dce, encoded_post_dce, delta_features_dce], dim=1)
        fused_features_dwi = torch.cat([encoded_pre_dwi, encoded_post_dwi, delta_features_dwi], dim=1)

        # 融合特征通过注意力模块
        attn_features_dce = self.fusion_module_dce(fused_features_dce)
        attn_features_dwi = self.fusion_module_dwi(fused_features_dwi)
        
        # 将两个模态的特征拼接起来进行辅助分类
        fused_all = torch.cat([attn_features_dce, attn_features_dwi], dim=1)

        aux_output = self.aux_classifier(fused_all)
        aux_loss = F.binary_cross_entropy(aux_output.squeeze(), pcr_label.float())
        
        # --- 6. 总损失 ---
        alpha = 1.0  # 重构损失权重
        beta = 1.0   # 对比损失权重
        gamma = 1.0  # 辅助分类损失权重

        total_loss = alpha * total_reco_loss + beta * contrast_loss + gamma * aux_loss
        
        return total_loss


# 示例用法：
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # 假设输入图像是 256x256，1 通道（灰度图）
    input_channels = 1 
    output_channels = 1 
    image_size = (224, 224) # 示例输入图像尺寸
    batch_size = 2

    # 创建虚拟输入图像
    dummy_pre_dce = torch.randn(batch_size, input_channels, image_size[0], image_size[1]).to(device)
    dummy_post_dce = torch.randn(batch_size, input_channels, image_size[0], image_size[1]).to(device)
    dummy_pre_dwi = torch.randn(batch_size, input_channels, image_size[0], image_size[1]).to(device)
    dummy_post_dwi = torch.randn(batch_size, input_channels, image_size[0], image_size[1]).to(device)
    label = torch.randint(0, 2, (batch_size,)).to(device)


    # 实例化模型
    num_down_blocks_in_encoder = 3 
    
    model = DoubleTower_Delta(in_channels=1,num_down_blocks=3).to(device)

    # 前向传播
    loss = model(dummy_pre_dce, dummy_post_dce, dummy_pre_dwi, dummy_post_dwi, label)

    print(f"原始治疗前图像形状: {loss}")

