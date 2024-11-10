'''
 # @ Author: Meet Patel
 # @ Create Time: 2024-11-07 10:32:00
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-11-09 09:07:17
 # @ Description:
 '''

import torch
import torch.nn as nn

# Define the generator network
class GeneratorNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(GeneratorNet, self).__init__()
        
        # Encoder layers
        self.e0 = nn.Conv2d(input_nc, ngf, kernel_size=5, stride=1, padding=2)
        self.e0_1 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf)
        )
        self.e1 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf)
        )
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 16)
        )
        # FIXME: Do we need 3 more conv with 1024 as per paper.

        # Decoder layers
        self.d4_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        self.d5_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        self.d6_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        self.d7_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf)
        )
        self.d8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf)
        )

        # Additional convolutional layers for the output
        self.output_layers = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # x : [1, 3, 256, 256]
        # Encoder forward pass
        e0 = self.e0(x)
        # [1, 64, 256, 256]
        
        e0_1 = self.e0_1(e0)
        # [1, 64, 256, 256]
        
        e1 = self.e1(e0_1)
        # [1, 64, 128, 128]
        
        e2 = self.e2(e1)
        # [1, 128, 64, 64]
        
        e3 = self.e3(e2)
        # [1, 256, 32, 32]
        
        e4 = self.e4(e3)
        # [1, 512, 16, 16]

        e5 = self.e5(e4)
        # [1, 1024, 8, 8]

        # Decoder forward pass with skip connections
        d4_ = self.d4_(e5)
        # [1, 512, 16, 16]

        d5_ = self.d5_(d4_ + e4)
        # [1, 256, 32, 32]

        d6_ = self.d6_(d5_ + e3)
        # [1, 128, 64, 64]

        d7_ = self.d7_(d6_ + e2)
        # [1, 64, 128, 128]
        
        d8 = self.d8(d7_ + e1)
        # [1, 64, 256, 256]

        # Output layers
        output = self.output_layers(d8)
        # [1, 3, 256, 256]
        
        return output


# Define the discriminator network
class DiscriminatorNet(nn.Module):
    def __init__(self, input_nc, output_nc, ndf):
        super(DiscriminatorNet, self).__init__()
        
        self.l1 = nn.Sequential(
            nn.Conv2d(input_nc + output_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.l2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.l3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.l4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.l6 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # x : [1, 6, 256, 256]
        
        l1 = self.l1(x)
        # [1, 48, 256, 256]

        l2 = self.l2(l1)
        # [1, 96, 256, 256]
        
        l3 = self.l3(l2)
        # [1, 192, 256, 256]

        l4 = self.l4(l3)
        # [1, 384, 256, 256]
        
        l6 = self.l6(l4)
        # [1, 1, 256, 256]
        
        return l6
    

if __name__ == "__main__":
    from torchsummary import summary
    input_nc = 3
    output_nc = 3
    input_hw = 256
    output_hw = 256
    base_channels = 64
    generator = GeneratorNet(input_nc=input_nc, output_nc=output_nc, ngf=base_channels)
    summary(generator, input_size=(input_nc, input_hw, input_hw), device="cpu")
    dummy_image = torch.randn(1, input_nc, input_hw, input_hw).to(torch.float32)
    gen_image = generator(dummy_image)
    assert list(gen_image.shape) == [1, output_nc, output_hw, output_hw]
    
    base_channels = 48
    discriminator = DiscriminatorNet(input_nc=input_nc, output_nc=output_nc, ndf=base_channels)
    dummy_image = torch.randn(1, input_nc + output_nc, input_hw, input_hw).to(torch.float32)
    summary(discriminator, input_size=(input_nc + output_nc, input_hw, input_hw), device="cpu")
    dis_output = discriminator(dummy_image)
    assert list(dis_output.shape) == [1, 1, output_hw, output_hw]
    