import torch
import torch.nn as nn
from att_modules import BandAtt


def conv3x3x3(in_channels, out_channels):
    return nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3),
                     stride=(1, 1, 1), padding=(1, 1, 1), bias=False)


class conv_prelu_res_prelu_block(nn.Module):
    def __init__(self):
        super(conv_prelu_res_prelu_block, self).__init__()
        self.conv1 = conv3x3x3(8, 8)
        self.prelu1 = nn.PReLU()
        self.conv2 = conv3x3x3(8, 8)
        self.prelu2 = nn.PReLU()
        self.se = BandAtt(8)
        # self.attn = Self_Attn(32)

    def forward(self, x):
        out = self.conv1(x)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.se(out)
        out = torch.add(out, x)
        out = self.prelu2(out)
        # out = self.attn(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, block_num):
        super(ResNet3D, self).__init__()

        self.input_conv = conv3x3x3(1, 8)
        self.conv_seq = self.make_layer(block, block_num)
        self.conv = conv3x3x3(8, 8)
        self.prelu = nn.PReLU()
        self.output_conv = conv3x3x3(8, 1)

    def make_layer(self, block, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(block())  # there is a ()
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.input_conv(x)
        residual = out
        out = self.conv_seq(out)
        out = self.conv(out)
        out = torch.add(out, residual)
        out = self.prelu(out)
        out = self.output_conv(out)
        return out


