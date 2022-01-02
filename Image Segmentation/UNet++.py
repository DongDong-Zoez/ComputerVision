#TO DO:

#1. Replace MaxPool2d by Conv2d with stride = 2
#2. Replace Upsample by ConvTranpose2d with stride = 2

from torch import nn
import torch

class DownSampleLayer(nn.Module):
    
    def __init__(self, in_ch):
        super(DownSampleLayer, self).__init__()
        
        self.Downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=in_ch),
            nn.ReLU()
        )
        
    def forward(self, x):
        
        x = self.Downsample(x)
        
        return x
    
class UpSampleLayer(nn.Module):
    
    def __init__(self, in_ch):
        super(UpSampleLayer, self).__init__()
        
        self.Upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=in_ch),
            nn.ReLU()
        )
        
    def forward(self, x):
        
        x = self.Upsample(x)
        
        return x
    
class DoubleConv(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        
        self.doubleConv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU()
        )
        

    def forward(self, x):
        
        x = self.doubleConv(x)
        
        return x

    
class UNetPlusPlus(nn.Module):
    
    def __init__(self, deep=4):
        super(UNetPlusPlus, self).__init__()

        def cat(TensorList, x):
            return torch.cat((*TensorList, x), dim=1)
        
        self.deep = deep      
        
        if self.deep not in range(1,5):
            return "The deep of network must be an integer in 1<=L<=4"
        
        in_channels = 1 #RGB
        out_channels= [32,64,128,256,512] #image tile
        
        self.down = nn.ModuleList([DownSampleLayer(out_channels[i]) for i in range(4)])
        self.up = nn.ModuleList([UpSampleLayer(out_channels[i+1]) for i in range(4)])
        
        self.cat = cat
        
        self.c0_0 = DoubleConv(in_channels, out_channels[0])
        self.c0_1 = DoubleConv(out_channels[0]+out_channels[1], out_channels[0])
        self.c0_2 = DoubleConv(out_channels[0]*2+out_channels[1], out_channels[0])
        self.c0_3 = DoubleConv(out_channels[0]*3+out_channels[1], out_channels[0])
        self.c0_4 = DoubleConv(out_channels[0]*4+out_channels[1], out_channels[0])
        
        self.c1_0 = DoubleConv(out_channels[0], out_channels[1])
        self.c1_1 = DoubleConv(out_channels[1]+out_channels[2], out_channels[1])
        self.c1_2 = DoubleConv(out_channels[1]*2+out_channels[2], out_channels[1])
        self.c1_3 = DoubleConv(out_channels[1]*3+out_channels[2], out_channels[1])
        
        self.c2_0 = DoubleConv(out_channels[1], out_channels[2])
        self.c2_1 = DoubleConv(out_channels[2]+out_channels[3], out_channels[2])
        self.c2_2 = DoubleConv(out_channels[2]*2+out_channels[3], out_channels[2])
        
        self.c3_0 = DoubleConv(out_channels[2], out_channels[3])
        self.c3_1 = DoubleConv(out_channels[3]+out_channels[4], out_channels[3])
        
        self.c4_0 = DoubleConv(out_channels[3], out_channels[4])
        
        self.output = nn.Conv2d(out_channels[0], 1, kernel_size=1, stride=1)

        
    def forward(self, x0_0):
        
        output = []
        
        x0_0 = self.c0_0(x0_0)
        x1_0 = self.c1_0(self.down[0](x0_0))
        x0_1 = self.c0_1(self.cat([x0_0], self.up[0](x1_0)))
        output.append(self.output(x0_1))
        
        if self.deep == 1:
            return sum(output) / len(output)
        
        x2_0 = self.c2_0(self.down[1](x1_0))
        x1_1 = self.c1_1(self.cat([x1_0], self.up[1](x2_0)))
        x0_2 = self.c0_2(self.cat([x0_0, x0_1], self.up[0](x1_1)))
        output2 = self.output(x0_2)
        
        if self.deep == 2:
            return sum(output) / len(output)
        
        x3_0 = self.c3_0(self.down[2](x2_0))
        x2_1 = self.c2_1(self.cat([x2_0], self.up[2](x3_0)))
        x1_2 = self.c1_2(self.cat([x1_0, x1_1], self.up[1](x2_1)))
        x0_3 = self.c0_3(self.cat([x0_0, x0_1, x0_2], self.up[0](x1_2)))
        output3 = self.output(x0_3)
        
        if self.deep == 3:
            return sum(output) / len(output)
        
        x4_0 = self.c4_0(self.down[3](x3_0))
        x3_1 = self.c3_1(self.cat([x3_0], self.up[3](x4_0)))
        x2_2 = self.c2_2(self.cat([x2_0, x2_1], self.up[2](x3_1)))
        x1_3 = self.c1_3(self.cat([x1_0, x1_1, x1_2], self.up[1](x2_2)))
        x0_4 = self.c0_4(self.cat([x0_0, x0_1, x0_2, x0_3], self.up[0](x1_3)))
        output4 = self.output(x0_4)
        
        if self.deep == 4:
            return sum(output) / len(output)

    def toDevice(self, device):
        for i in range(len(self.up)):
            self.up[i].to(device)
            self.down[i].to(device)
