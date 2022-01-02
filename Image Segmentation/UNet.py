from torch import nn
import torch

class DownSampleLayer(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(DownSampleLayer, self).__init__()
        
        self.DoubleConv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU()
        )
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2)
        )
        
    def forward(self, x):
        
        x = self.DoubleConv(x)
        d = self.downsample(x)
        
        return x, d
    
class UpSampleLayer(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(UpSampleLayer, self).__init__()
        
        self.DoubleConv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_ch*2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_ch*2),
            nn.ReLU()
        )
        
        self.Upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2, out_channels=out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU()
        )
    
    def forward(self, x, copy_crop):
        
        x = self.DoubleConv(x)
        u = self.Upsample(x)
        copy_crop = torch.cat((u, copy_crop), dim=1)
        
        return copy_crop
    
class UNet(nn.Module):
    
    def __init__(self):
        super(UNet, self).__init__()
        
        in_channels = 1 #RGB
        out_channels= [16,32,64,128,256] #image tile
        
        self.d1 = DownSampleLayer(in_ch=in_channels, out_ch=out_channels[0])
        self.d2 = DownSampleLayer(in_ch=out_channels[0], out_ch=out_channels[1])
        self.d3 = DownSampleLayer(in_ch=out_channels[1], out_ch=out_channels[2])
        self.d4 = DownSampleLayer(in_ch=out_channels[2], out_ch=out_channels[3])
        
        self.u1 = UpSampleLayer(in_ch=out_channels[3], out_ch=out_channels[3])
        self.u2 = UpSampleLayer(in_ch=out_channels[4], out_ch=out_channels[2])
        self.u3 = UpSampleLayer(in_ch=out_channels[3], out_ch=out_channels[1])
        self.u4 = UpSampleLayer(in_ch=out_channels[2], out_ch=out_channels[0])
        
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels[0]),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels[0]),
            nn.ReLU(),    
            
            nn.Conv2d(in_channels=out_channels[0], out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
            
        c1, d1 = self.d1(x)
        c2, d2 = self.d2(d1)
        c3, d3 = self.d3(d2)
        c4, d4 = self.d4(d3)
            
        u1 = self.u1(d4, c4)
        u2 = self.u2(u1, c3)
        u3 = self.u3(u2, c2)
        u4 = self.u4(u3, c1)
            
        out = self.output(u4)
            
        return out
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        if not cpu:
            self.load_state_dict(torch.load(filename))
        else:
            self.__init__(self.nbase,
                    self.nout,
                    self.kernel_size,
                    self.concatenation)

            self.load_state_dict(torch.load(filename,
                                      map_location=torch.device('cpu')))