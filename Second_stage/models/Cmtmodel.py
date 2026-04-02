import torch
import torch.nn as nn
import os # For loading weights

# =========================================================================
# 以下是你提供的原始辅助模块和融合模块代码，我将它们放在这里以保持完整性
# 请确保这些类在你的实际代码中是可用的，例如 GELU(), Encoder(), MyDataSet, get_train_transforms, get_test_transforms 等
# 如果某个类定义缺失，你需要自行补充或调整导入路径。
# 为了代码可运行，我在这里假设 GELU 是已定义的激活函数。


class CMT(nn.Module):
    def __init__(self, in_channels): # 增加 in_channels 参数
        super(CMT, self).__init__()
        # 调整通道数以匹配 Encoder 的输出 (256)
        self.channel_conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.channel_conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)

        self.spatial_conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.spatial_conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.conv11 = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1, padding=0) # 拼接后是 2*in_channels

    def forward(self, query, key):
        ######################################## channel
        chn_key = self.channel_conv_1(key)  # B * 1 * H * W
        chn_query = self.channel_conv_2(query)  # B * C * H * W

        B, C, H, W = chn_query.size()

        chn_query_unfold = chn_query.view(B, C, H * W)  # B * C * (HW)
        chn_key_unfold = chn_key.view(B, 1, H * W)  # B * 1 * (HW)

        chn_key_unfold = chn_key_unfold.permute(0, 2, 1) # B * (HW) * 1

        # chn_query_relevance: [B, C, HW] @ [B, HW, 1] -> [B, C, 1]
        chn_query_relevance = torch.bmm(chn_query_unfold, chn_key_unfold) 
        chn_query_relevance_ = torch.sigmoid(chn_query_relevance) 
        chn_query_relevance_ = 1 - chn_query_relevance_ #irrelevance map(channel)
        inv_chn_query_relevance_ = chn_query_relevance_.unsqueeze(3) # B * C * 1 * 1
        chn_value_final = inv_chn_query_relevance_ * query # 广播乘法

        ######################################## spatial
        spa_key = self.spatial_conv_1(key)  # B * C * H * W
        spa_query = self.spatial_conv_2(query)  # B * C * H * W

        B, C, H, W = spa_query.size()

        spa_query_unfold = spa_query.view(B, H * W, C)  # B * (HW) * C
        spa_key_unfold = spa_key.view(B, H * W, C)  # B * (HW) * C

        spa_key_unfold = torch.mean(spa_key_unfold, dim=1) # B * C
        spa_key_unfold = spa_key_unfold.unsqueeze(2) # B * C * 1

        # spa_query_relevance: [B, HW, C] @ [B, C, 1] -> [B, HW, 1]
        spa_query_relevance = torch.bmm(spa_query_unfold, spa_key_unfold)
        spa_query_relevance = torch.sigmoid(spa_query_relevance) 

        inv_spa_query_relevance = 1 - spa_query_relevance #irrelevance map(spatial)
        inv_spa_query_relevance_ = inv_spa_query_relevance.permute(0, 2, 1) # B * 1 * HW
        inv_spa_query_relevance_ = inv_spa_query_relevance_.view(B, 1, H, W)
        spa_value_final = inv_spa_query_relevance_ * query # 广播乘法

        key_relevance = torch.cat([chn_value_final, spa_value_final], dim =1)
        key_relevance = self.conv11(key_relevance)

        return key_relevance

class CMT_transformers(nn.Module):
    def __init__(self, in_channels): # 传入 in_channels
        super(CMT_transformers, self).__init__()
        # 调整通道数以匹配 Encoder 的输出 (256)
        self.bot_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bot_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.fusionTransformer_vis = CMT(in_channels) # 传入 in_channels
        self.fusionTransformer_ir = CMT(in_channels) # 传入 in_channels

    def forward(self, rgb, ir): # 这里 rgb 对应 DCE 特征，ir 对应 DWI 特征
        # gated_bottleneck
        bot_feature = rgb + ir
        bot_rgb = self.bot_conv1(bot_feature)
        bot_rgb_ = torch.sigmoid(bot_rgb)

        bot_ir = self.bot_conv2(bot_feature)
        bot_ir_ = torch.sigmoid(bot_ir)

        # normalization
        # 注意：这里可能存在除以零的风险，如果 bot_rgb_ + bot_ir_ 接近于零。
        # 实际应用中可能需要加一个小的 epsilon
        sum_bottleneck = bot_rgb_ + bot_ir_ + 1e-6 
        bot_rgb_ = bot_rgb_ / sum_bottleneck
        bot_ir_ = bot_ir_ / sum_bottleneck

        # transformer
        rgb_hat = self.fusionTransformer_vis(rgb, ir * bot_ir_) # query=rgb, key=ir*bot_ir_
        ir_hat = self.fusionTransformer_ir(ir, rgb * bot_rgb_) # query=ir, key=rgb*bot_rgb_

        return rgb_hat, ir_hat


# =========================================================================

# 修改后的 MedicalImageProcessor，使用 CMT_transformers
class MedicalImageProcessor_CMT(nn.Module):
    def __init__(self, in_channel=1, d_model=256, dropout=0.25, num_heads=8, encoder_out_channels=256, weights_path=None, device=torch.device("cuda")):
        super().__init__()
        self.device = device
        self.weights_path = weights_path
        
        # Encoder编码器
        self.img_encoder_dce = Encoder(in_channels=in_channel)
        self.img_encoder_dwi = Encoder(in_channels=in_channel)
        
        # ---- 权重加载 (保持不变) ----
        if weights_path:
            self._load_and_freeze_weights()

        # CMT_transformers 融合模块
        # 注意：这里的 encoder_out_channels 对应特征图的通道数
        self.cmt_fusion = CMT_transformers(in_channels=encoder_out_channels)

        # 全局池化层，用于将融合后的特征图转换为向量
        self.global_pool = nn.AdaptiveAvgPool2d(1) # 使用 AdaptiveAvgPool2d 因为输入是 2D 特征图

        # 投影层，将融合后的特征向量投影到最终的 d_model 维度
        # CMT_transformers 的输出是两个 [B, C, H, W] 的特征图，我们将它们拼接后池化
        # 那么拼接后的通道数是 encoder_out_channels * 2
        self.proj = nn.Sequential(
            nn.Linear(encoder_out_channels * 2, d_model), # 两个模态的特征拼接后，通道翻倍
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout)
        )

    def _load_and_freeze_weights(self):
        """权重加载与冻结专用方法"""
        # 加载权重
        # 确保这个路径是正确的，并且权重文件存在
        # weights_path = '/home/zhoutz/MY_Project/First_stage_2d/model_weight/best_model_fold1.pth'
        if not os.path.exists(self.weights_path):
            print(f"警告: 权重文件 {self.weights_path} 不存在。Encoder 将使用随机初始化权重。")
            return

        device = next(self.parameters()).device
        weights = torch.load(self.weights_path, map_location=device)
        
        # DCE Encoder
        dce_weights = {k.replace('img_encoder_dce.', ''): v 
                       for k, v in weights.items() 
                       if k.startswith('img_encoder_dce.')}
        missing_dce, _ = self.img_encoder_dce.load_state_dict(dce_weights, strict=False)
        
        # DWI Encoder
        dwi_weights = {k.replace('img_encoder_dwi.', ''): v 
                       for k, v in weights.items() 
                       if k.startswith('img_encoder_dwi.')}
        missing_dwi, _ = self.img_encoder_dwi.load_state_dict(dwi_weights, strict=False)
        
        # 冻结 Encoder 权重 (如果需要)
        # for param in self.img_encoder_dce.parameters():
        #     param.requires_grad = False
        # for param in self.img_encoder_dwi.parameters():
        #     param.requires_grad = False
        # print("Encoder 权重已冻结.")

    def forward(self, pre_dce, pre_dwi):
        """
        Args:
            pre_dce: DCE 图像输入, [B, C_in, H, W]
            pre_dwi: DWI 图像输入, [B, C_in, H, W]
        Returns:
            img_feat: 融合并投影后的特征向量 [B, d_model]
            dce_feat_map: 原始 DCE 特征图 [B, encoder_out_channels, H_out, W_out]
            dwi_feat_map: 原始 DWI 特征图 [B, encoder_out_channels, H_out, W_out]
        """
        # MRI图像处理 - 提取特征图
        dce_feat_map = self.img_encoder_dce(pre_dce) # 假设输出 [B, 256, 32, 32]
        dwi_feat_map = self.img_encoder_dwi(pre_dwi) # 假设输出 [B, 256, 32, 32]

        # 使用 CMT_transformers 进行特征图级别的融合
        # 得到两个融合后的特征图，分别代表对原模态的增强
        fused_dce_feat_map, fused_dwi_feat_map = self.cmt_fusion(dce_feat_map, dwi_feat_map)

        # 将融合后的两个特征图拼接起来
        # [B, 256, H, W] 和 [B, 256, H, W] 拼接 -> [B, 512, H, W]
        combined_fused_map = torch.cat((fused_dce_feat_map, fused_dwi_feat_map), dim=1)

        # 对拼接后的特征图进行全局池化，转换为向量
        pooled_fused_vector = self.global_pool(combined_fused_map).flatten(start_dim=1) # [B, 512]

        # 投影到最终的 d_model 维度
        img_feat = self.proj(pooled_fused_vector) # [B, d_model]

        return img_feat, dce_feat_map, dwi_feat_map # 也可以返回融合后的特征图 fused_dce_feat_map, fused_dwi_feat_map
    
