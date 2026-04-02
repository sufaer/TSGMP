import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, dilation=1, is_last=False):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        self.groups = groups

        # 调整卷积层的输入/输出通道，以处理傅里叶变换后的实部和虚部
        # 傅里叶变换后，通道会变成 in_channels * 2 (实部和虚部)
        self.conv_layer_vis = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                            out_channels=out_channels * 2, # 输出也是实部和虚部
                                            kernel_size=3, stride=1, padding=1, groups=self.groups, bias=False)
        self.conv_layer_ir = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                            out_channels=out_channels * 2, # 输出也是实部和虚部
                                            kernel_size=3, stride=1, padding=1, groups=self.groups, bias=False)

        # 拼接后通道是 out_channels * 4，降维到 out_channels * 2 (实部和虚部)
        self.conv = ConvLayer(out_channels * 4, out_channels * 2, kernel_size=1, stride=1, padding=0)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

        self.Prelu1 = nn.PReLU()
        self.Prelu2 = nn.PReLU()

    def forward(self, vis, ir): # 这里 vis 对应 DCE 特征，ir 对应 DWI 特征
        batch = vis.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = vis.shape[-2:]
            vis = F.interpolate(vis, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)
            ir = F.interpolate(ir, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False) # ir 也需要插值

        # 傅里叶变换
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted_vis = torch.fft.rfftn(vis, dim=fft_dim, norm=self.fft_norm)
        ffted_vis = torch.stack((ffted_vis.real, ffted_vis.imag), dim=-1)
        ffted_vis = ffted_vis.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted_vis = ffted_vis.view((batch, -1,) + ffted_vis.size()[3:]) # (batch, c*2, h, w/2+1)

        ffted_ir = torch.fft.rfftn(ir, dim=fft_dim, norm=self.fft_norm)
        ffted_ir = torch.stack((ffted_ir.real, ffted_ir.imag), dim=-1)
        ffted_ir = ffted_ir.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted_ir = ffted_ir.view((batch, -1,) + ffted_ir.size()[3:]) # (batch, c*2, h, w/2+1)

        # 在频域进行卷积
        ffted_vis_conv = self.conv_layer_vis(ffted_vis)  # (batch, out_c*2, h, w/2+1)
        ffted_ir_conv = self.conv_layer_ir(ffted_ir)      # (batch, out_c*2, h, w/2+1)
        
        # PReLU 应用在独立的特征上
        ffted_vis_conv = self.Prelu1(ffted_vis_conv)
        ffted_ir_conv = self.Prelu2(ffted_ir_conv) # 注意这里你原代码是 ffted_ir = self.Prelu1(ffted_vis) ffted_ir = self.Prelu2(ffted_ir)
                                                    # 这意味着 Prelu2 应用在了 ffted_vis 上，然后又应用在了 ffted_ir 上
                                                    # 我这里修改为 Prelu1 应用在 vis 的 conv 结果， Prelu2 应用在 ir 的 conv 结果
                                                    # 如果你原意是 Prelu1 和 Prelu2 对同一个模态的结果进行多次激活，请自行调整
                                                    # 假设是处理两个独立路径的激活

        # 拼接融合后的频域特征
        ffted_fused = torch.cat([ffted_vis_conv, ffted_ir_conv], dim=1) # (batch, out_c*4, h, w/2+1)
        ffted_fused = self.conv(ffted_fused) # (batch, out_c*2, h, w/2+1)

        # 逆傅里叶变换前的数据重塑
        ffted_fused = ffted_fused.view((batch, -1, 2,) + ffted_fused.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch, out_c, h, w/2+1, 2)
        ffted_fused_complex = torch.complex(ffted_fused[..., 0], ffted_fused[..., 1])

        # 逆傅里叶变换回空域
        ifft_shape_slice = vis.shape[-3:] if self.ffc3d else vis.shape[-2:]
        output = torch.fft.irfftn(ffted_fused_complex, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output

class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, **fu_kwargs):
        super(SpectralTransform, self).__init__()
        self.fu = FourierUnit(in_channels, out_channels, groups, **fu_kwargs)

    def forward(self, vis, ir):
        output = self.fu(vis, ir)
        return output

