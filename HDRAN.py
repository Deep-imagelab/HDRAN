import torch
import torch.nn as nn
# from math import sqrt
from att_modules import ChannelAtt
from utils import conv_prelu_res_prelu_block, ResNet3D


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=True)

    
class conv_relu_res_relu_block(nn.Module):
    def __init__(self):
        super(conv_relu_res_relu_block, self).__init__()
        self.conv1 = conv3x3(256, 256)
        self.relu1 = nn.PReLU()
        self.conv2 = conv3x3(256, 256)
        self.relu2 = nn.PReLU()
        self.se = ChannelAtt(256, 16)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.se(out)
        out = torch.add(out,residual) 
        out = self.relu2(out)
        return out

    
class HDRAN(nn.Module):
    def __init__(self, block, block_num, input_channel, output_channel):
        super(HDRAN, self).__init__()

        self.in_channels = input_channel
        self.out_channels = output_channel
        self.input_conv = conv3x3(self.in_channels, out_channels=256)  
        self.conv_seq = self.make_layer(block, block_num)
        self.conv = conv3x3(256, 256)
        self.relu = nn.PReLU()
        self.output_conv = conv3x3(in_channels=256,  out_channels=self.out_channels)
        self.second_network = ResNet3D(conv_prelu_res_prelu_block, 4)

    def make_layer(self,block,num_layers):
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
        out = self.relu(out)
        out = self.output_conv(out)
        out = out.unsqueeze(1)
        out = self.second_network(out)
        out = out.squeeze(1)
        return out


if __name__ == '__main__':
    input_tensor = torch.randn(2, 3, 64, 64)
    model = HDRAN(conv_relu_res_relu_block, 16, 3, 31)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    print(output_tensor.shape)
