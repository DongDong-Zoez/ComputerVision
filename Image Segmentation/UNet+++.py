import torch 
from torch import nn

class UpsampleLayer(nn.Module):
    
    def __init_bilinear(self, scale_factor=None):
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=self.mode)
        )
    
    def __init_deconv(self, in_channels=None, out_channels=None):
        self.up = nn.Sequential(
            nn.ConvTranpose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def __init__(self, mode='bilinear', **kwargs):
        super(UpsampleLayer, self).__init__()
        
        self.mode = mode
        if self.mode not in ['bilinear', 'deconv']:
            raise Exception("mode argument must be \'bilinear\' or \'deconv\'")
        elif self.mode == 'bilinear':
            self.__init_bilinear(**kwargs)
        elif self.mode == 'deconv':
            self.__init_deconv(**kwargs)

        
    def forward(self, x):
        
        x = self.up(x)
        
        return x
    
class DownsampleLayer(nn.Module):
    
    def __init_pooling(self, stride=None):
        self.down = nn.Sequential(
            nn.MaxPool2d(stride),
        )
    
    def __init_conv(self, in_channels=None, out_channels=None):
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
      
    def __init__(self, mode='maxpooling', **kwargs):
        super(DownsampleLayer, self).__init__()
        
        self.mode = mode
        if self.mode not in ['maxpooling', 'conv']:
            raise Exception("mode argument must be \'maxpooling\' or \'conv\'")
        elif self.mode == 'maxpooling':
            self.__init_pooling(**kwargs)
        elif self.mode == 'conv':
            self.__init_conv(**kwargs)
            
    def forward(self, x):
        
        x = self.down(x)
        
        return x
    
class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DoubleConv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x
    
class fullscale_connection(nn.Module):
    
    def __init__(self, mode, scale_factor, in_channels, out_channels, kernel_size, stride, padding):
        super(fullscale_connection, self).__init__()
        self.mode = mode
        self.connection = nn.Sequential()
        if self.mode == 'up':
            self.connection.add_module(f'Up{scale_factor}', UpsampleLayer(mode='bilinear', scale_factor=scale_factor))
            self.connection.add_module('Conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        elif self.mode == 'down':
            self.connection.add_module(f'Down{scale_factor}', DownsampleLayer(mode='maxpooling', stride=scale_factor))
            self.connection.add_module('Conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        elif self.mode == 'direct':
            self.connection.add_module('Conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            
    def forward(self, x):
        
        x = self.connection(x)

        return x
    
class FullScaleUNet(nn.Module):
    
    def __init__(self, num_encoders=5):
        super(FullScaleUNet, self).__init__()
        
        out_channels= [3,64,128,256,512,1024]
        self.sigmoid = nn.Sigmoid()
        
        self.Encoder = nn.ModuleList(
            [DoubleConv(out_channels[i], out_channels[i+1], kernel_size=3, stride=1, padding=1) for i in range(num_encoders)]
        )
        
        self.Decoder = DoubleConv(out_channels[1]*num_encoders, out_channels[1]*num_encoders, 
                                  kernel_size=3, stride=1, padding=1)
        
        self.down = DownsampleLayer(mode='maxpooling', stride=2)
        self.up = nn.ModuleList([UpsampleLayer(mode='bilinear', scale_factor=2**(i+1)) for i in range(num_encoders-1)])
        
        self.inter = nn.ModuleList([nn.ModuleList(
                                   [fullscale_connection(mode='down', scale_factor=2**(j+1), in_channels=out_channels[i+1], 
                                    out_channels=out_channels[1], kernel_size=3, stride=1, padding=1) 
                                    for i in range(num_encoders)]) for j in range(num_encoders-1)])

        self.intra = nn.ModuleList([fullscale_connection(mode='up', scale_factor=2**(j+1), 
                                    in_channels=out_channels[1]*num_encoders, out_channels=out_channels[1], kernel_size=3, 
                                    stride=1, padding=1) for j in range(num_encoders-1)])
        
        self.transition = nn.ModuleList([fullscale_connection(mode='up', scale_factor=2**(j+1),
                                         in_channels=out_channels[num_encoders], out_channels=out_channels[1],
                                         kernel_size=3, stride=1, padding=1) for j in range(num_encoders-1)])
        
        self.direct = nn.ModuleList([fullscale_connection(mode='direct', scale_factor=1, in_channels=out_channels[i+1],
                                     out_channels=out_channels[1], kernel_size=3, stride=1, padding=1)
                                     for i in range(num_encoders)])
        
        self.En_supervise = nn.Conv2d(out_channels[num_encoders], 3, kernel_size=3, stride=1, padding=1)
        self.De_supervise = nn.Conv2d(out_channels[1]*num_encoders, 3, kernel_size=3, stride=1, padding=1)

        
    def forward(self, x):
        
        e0 = self.Encoder[0](x)
        e1 = self.Encoder[1](self.down(e0))
        e2 = self.Encoder[2](self.down(e1))
        e3 = self.Encoder[3](self.down(e2))
        e4 = self.Encoder[4](self.down(e3))
    
        d3 = self.Decoder(torch.cat([self.transition[0](e4), self.direct[3](e3), self.inter[0][2](e2),
                                     self.inter[1][1](e1), self.inter[2][0](e0)], dim=1))
        d2 = self.Decoder(torch.cat([self.transition[1](e4), self.intra[0](d3), self.direct[2](e2),
                                     self.inter[0][1](e1), self.inter[1][0](e0)], dim=1))
        d1 = self.Decoder(torch.cat([self.transition[2](e4), self.intra[1](d3), self.intra[0](d2),
                                     self.direct[1](e1), self.inter[0][0](e0)], dim=1))
        d0 = self.Decoder(torch.cat([self.transition[3](e4), self.intra[2](d3), self.intra[1](d2),
                                     self.intra[0](d1), self.direct[0](e0)], dim=1))
        
        s4 = self.En_supervise(e4)
        s4 = self.up[3](s4)
        s4 = self.sigmoid(s4)
        
        s3 = self.De_supervise(d3)
        s3 = self.up[2](s3)
        s3 = self.sigmoid(s3)
        
        s2 = self.De_supervise(d2)
        s2 = self.up[1](s2)
        s2 = self.sigmoid(s2)
        
        s1 = self.De_supervise(d1)
        s1 = self.up[0](s1) 
        s1 = self.sigmoid(s1)
        
        s0 = self.De_supervise(d0)
        s0 = self.sigmoid(s0)
        
        return s4, s3, s2, s1, s0
