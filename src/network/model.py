import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
        ]
        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.residual(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define your generator architecture
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        for _ in range(3):  # Add residual blocks
            self.model.add_module("residual", ResidualBlock(256, 256))

        self.model.add_module(
            "deconv1",
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.model.add_module("instnorm1", nn.InstanceNorm2d(128))
        self.model.add_module("relu1", nn.ReLU(inplace=True))

        self.model.add_module(
            "deconv2",
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.model.add_module("instnorm2", nn.InstanceNorm2d(64))
        self.model.add_module("relu2", nn.ReLU(inplace=True))

        self.model.add_module("conv2d_out", nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=3))
        self.model.add_module("tanh", nn.Tanh())

    def forward(self, x):
        return self.model(x)


# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Define your discriminator architecture
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
        )

    def forward(self, x):
        return self.model(x)
