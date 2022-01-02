class ConvLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, num_conv):
        super(ConvLayer, self).__init__()
        
        def convReLU(self, in_channels, out_channels, kernel_size):
            return nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                        #nn.BatchNorm2d(out_channels, eps=1e-5),
                        nn.ReLU(inplace=True),
                  )
        
        self.conv =  nn.Sequential()
        for i in range(num_conv):
            if i == 0:
                self.conv.add_module('conv_%d'%i, convReLU(self, in_channels, out_channels, kernel_size))
            else:
                self.conv.add_module('conv_%d'%i, convReLU(self, out_channels, out_channels, kernel_size))
                
        self.conv.add_module('MaxPooling', nn.MaxPool2d(2))
        
    def forward(self, x):
        return self.conv(x)
    
class VGG16(nn.Module):
    
    def __init__(self):
        super(VGG16, self).__init__()
        
        in_channels=3
        out_channels=[64, 128, 256, 512, 512]
        num_conv = [2, 2, 3, 3, 3]
        kernel_size=3
        dense_len = [4096, 4096]
        classes = 2
        
        
        self.l1 = ConvLayer(in_channels, out_channels[0], kernel_size, num_conv[0])
        self.l2 = ConvLayer(out_channels[0], out_channels[1], kernel_size, num_conv[1])
        self.l3 = ConvLayer(out_channels[1], out_channels[2], kernel_size, num_conv[2])
        self.l4 = ConvLayer(out_channels[2], out_channels[3], kernel_size, num_conv[3])
        self.l5 = ConvLayer(out_channels[3], out_channels[4], kernel_size, num_conv[4])
        
        self.avepool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        
        self.dense = nn.Sequential(
            nn.Linear(7*7*512, dense_len[0]),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(dense_len[0], dense_len[1]),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(dense_len[1], classes),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.avepool(x)
        x = x.view(x.size(0), -1) # x.size(0) = batch_size
        x = self.dense(x)
        
        return x
