"""
Networks created by Andres Diaz-Pinto
"""

import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self, out_size):
        super(Flatten, self).__init__()
        self.out_size = out_size
    def forward(self, input):
        return input.view(-1, self.out_size)

class LinearDec(nn.Module):
    def __init__(self, out_size):
        super(LinearDec, self).__init__()
        self.out_size = out_size
    def forward(self, input):
        return input.view(-1, self.out_size[0], self.out_size[1], self.out_size[2])


################# NETS FOR CMR IMAGES

class encoder_cmr(nn.Module):
    def __init__(self, image_channels, ndf, z_dim):
        super(encoder_cmr, self).__init__()

        self.image_channels = image_channels
        self.ndf = ndf
        self.z_dim = z_dim

        self.encoder = nn.Sequential(

            nn.Conv2d(self.image_channels, self.ndf, 4, 2, 1),
            nn.BatchNorm2d(self.ndf),
            nn.ReLU(),

            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*2),
            nn.ReLU(),

            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*4),
            nn.ReLU(),

            nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*8),
            nn.ReLU(),

            nn.Conv2d(self.ndf*8, self.ndf*16, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*16),
            nn.ReLU(),

            Flatten(out_size = self.ndf*16*4*4),

            nn.Linear(self.ndf*16*4*4, self.z_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


class decoder_cmr(nn.Module):
    def __init__(self, image_channels, ndf, z_dim):
        super(decoder_cmr, self).__init__()

        self.image_channels = image_channels
        self.ndf = ndf
        self.z_dim = z_dim

        self.decoder = nn.Sequential(

            nn.Linear(self.z_dim, self.ndf*16*4*4),
            nn.LeakyReLU(0.2),

            LinearDec(out_size = [self.ndf*16, 4, 4]),

            nn.ConvTranspose2d(self.ndf*16, self.ndf*8, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(self.ndf*8, self.ndf*4, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(self.ndf*4, self.ndf*2, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(self.ndf*2, self.ndf, 4, 2, 1),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(self.ndf, self.image_channels, 4, 2, 1),
            nn.Tanh(),
            )

    def forward(self, x):
        h = self.decoder(x)
        return h


if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # 假设输入图像是 256x256，1 通道（灰度图）
    input_channels = 1 
    output_channels = 1 
    image_size = (256, 256) # 示例输入图像尺寸
    batch_size = 1

    # 创建虚拟输入图像
    dummy_pre_dce = torch.randn(batch_size, input_channels, image_size[0], image_size[1]).to(device)
    dummy_post_dce = torch.randn(batch_size, input_channels, image_size[0], image_size[1]).to(device)
    dummy_pre_dwi = torch.randn(batch_size, input_channels, image_size[0], image_size[1]).to(device)
    dummy_post_dwi = torch.randn(batch_size, input_channels, image_size[0], image_size[1]).to(device)
    label = torch.randint(0, 2, (batch_size,)).to(device)


    # 实例化模型
    num_down_blocks_in_encoder = 3 
    
    encoder = encoder_cmr(image_channels=1, ndf=64, z_dim=128).to(device)

    # # 逐层维度跟踪
    # def print_layer_shapes(model, input_tensor):
    #     hooks = []
    #     def hook_fn(module, input, output):
    #         print(f"{module.__class__.__name__}: {output.shape}")
        
    #     for layer in model.encoder:
    #         hooks.append(layer.register_forward_hook(hook_fn))
        
    #     _ = model(input_tensor)
    #     [h.remove() for h in hooks]

    # print_layer_shapes(encoder, dummy_pre_dce)



    decoder = decoder_cmr(image_channels=1, ndf=64, z_dim=128).to(device)
    z = encoder(dummy_pre_dce)  # 获取潜在编码

    # 逆向维度检查
    def print_decoder_shapes(model, input_tensor):
        hooks = []
        def hook_fn(module, input, output):
            print(f"{module.__class__.__name__}: {output.shape}")
        
        for layer in model.decoder:
            hooks.append(layer.register_forward_hook(hook_fn))
        
        _ = model(input_tensor)
        [h.remove() for h in hooks]

    print_decoder_shapes(decoder, z)


