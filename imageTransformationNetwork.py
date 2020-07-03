import torch
import torch.nn as nn
import numpy as np

class Normalization(nn.Module):
    def __init__(self, n_channels, norm):
        super(Normalization, self).__init__()
        if norm=='instance':
            self.normalization = nn.InstanceNorm2d(n_channels, affine=True)
        else:
            self.normalization = nn.BatchNorm2d(n_channels, affine=True)

    def forward(self, x):
        return self.normalization(x)

class ConvLayer(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.reflectionPad = nn.ReflectionPad2d(int(np.floor(kernel_size/2)))
        self.conv2d = nn.Conv2d(input_ch, output_ch, kernel_size, stride)

    def forward(self, x):
        out = self.reflectionPad(x)
        return self.conv2d(out)


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.conv_1 = ConvLayer(n_channels, n_channels, kernel_size=3, stride=1)
        self.in_norm = Normalization(n_channels, norm='instance')
        self.conv_2 = ConvLayer(n_channels, n_channels, kernel_size=3, stride=1)
        self.in_norm_2 = Normalization(n_channels, norm='instance')
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in_norm(self.conv_1(x)))
        out = self.in_norm_2(self.conv_2(out))
        return out + residual

class UpSampleConvLayer(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride, upsample=None):
        super(UpSampleConvLayer, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.upsampleLayer = nn.Upsample(scale_factor=self.upsample)
        self.reflectionPad = nn.ReflectionPad2d(int(np.floor(kernel_size/2)))
        self.conv2d = nn.Conv2d(input_ch, output_ch, kernel_size, stride)

    def forward(self, x):
        input = x
        if self.upsample:
            input = self.upsampleLayer(x)
        out = self.reflectionPad(input)
        return self.conv2d(out)

class ImageTransformationNetwork(nn.Module):
    def __init__(self):
        super(ImageTransformationNetwork, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            Normalization(32, norm='instance'),
            nn.ReLU(),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            Normalization(64, norm='instance'),
            nn.ReLU(),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            Normalization(128, norm='instance'),
            nn.ReLU(),
        )
        # self.conv_1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        # self.norm1 = Normalization(32, norm='instance')
        # self.conv_2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        # self.norm2 = Normalization(64, norm='instance')
        # self.conv_3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        # self.norm3 = Normalization(128, norm='instance')

        self.ResidualLayer = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.DeConvBlock = nn.Sequential(
            UpSampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
            Normalization(64, norm='instance'),
            nn.ReLU(),
            UpSampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            Normalization(32, norm='instance'),
            nn.ReLU(),
            ConvLayer(32, 3, kernel_size=9, stride=1)
        )
        # self.conv_4 = UpSampleConvLayer(128, 64, kernel_size=3, stride=1/2, upsample=2)
        # self.conv_5 = UpSampleConvLayer(64, 32, kernel_size=3, stride=1/2, upsample=2)


    def forward(self, x):
        out = self.ConvBlock(x)
        out = self.ResidualLayer(out)
        return self.DeConvBlock(out)
