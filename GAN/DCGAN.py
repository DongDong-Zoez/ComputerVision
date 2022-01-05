from torch import nn

class discriminator(nn.Module):
    
    def __init__(self, kernel_size=4, stride=2, padding=1, p=0.2, num_conv=5):
        super(discriminator, self).__init__()
        self.ConvLayer = nn.Sequential()

        in_channels = 3
        out_channels = [32, 64, 128, 256, 512]

        for i in range(num_conv):
            if i == 0:
                self.ConvLayer.add_module(f"Conv_{i}", nn.Conv2d(in_channels, out_channels[i], kernel_size, stride, padding, bias=False))
                #self.ConvLayer.add_module(f"BatchNorm{i}", nn.BatchNorm2d(out_channels[i]))
                self.ConvLayer.add_module(f"LeakyReLU_{i}", nn.LeakyReLU(p, inplace=True))
            else:
                self.ConvLayer.add_module(f"Conv_{i}", nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size, stride, padding, bias=False))
                self.ConvLayer.add_module(f"BatchNorm{i}", nn.BatchNorm2d(out_channels[i]))
                self.ConvLayer.add_module(f"LeakyReLU_{i}", nn.LeakyReLU(p, inplace=True))

        self.classifier = nn.Sequential()
        self.classifier.add_module("FinalConv_cls", nn.Conv2d(out_channels[num_conv-1], 1, kernel_size, stride=1, padding=0, bias=False))
        self.classifier.add_module("Sigmoid_cls", nn.Sigmoid())


    def forward(self, x):
        
        x = self.ConvLayer(x)
        x = self.classifier(x)
        
        return x

class generator(nn.Module):
   
    def __init__(self, kernel_size=4, stride=2, padding=1, num_conv=5):
        super(generator, self).__init__()
        self.ConvLayer = nn.Sequential()

        in_channels = 100
        out_channels = [1024, 512, 256, 128, 64]

        for i in range(num_conv):
            if i == 0:
                self.ConvLayer.add_module(f"DeConv_{i}", nn.ConvTranspose2d(in_channels, out_channels[i], kernel_size, stride=1, padding=0, bias=False))
                #self.ConvLayer.add_module(f"BatchNorm{i}", nn.BatchNorm2d(out_channels[i]))
                self.ConvLayer.add_module(f"ReLU_{i}", nn.ReLU(inplace=True))
            else:
                self.ConvLayer.add_module(f"DeConv_{i}", nn.ConvTranspose2d(out_channels[i-1], out_channels[i], kernel_size, stride, padding, bias=False))
                self.ConvLayer.add_module(f"BatchNorm{i}", nn.BatchNorm2d(out_channels[i]))
                self.ConvLayer.add_module(f"ReLU_{i}", nn.ReLU(inplace=True))

        self.classifier = nn.Sequential()
        self.classifier.add_module("FinalDeConv_cls", nn.ConvTranspose2d(out_channels[num_conv-1], 3, kernel_size, stride, padding, bias=False))
        self.classifier.add_module("Tanh_cls", nn.Tanh())


    def forward(self, x):
        
        x = self.ConvLayer(x)
        x = self.classifier(x)
        
        return x
