import torch.nn as nn
import torch

class AlexNet_GAP(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet_GAP, self).__init__()
        
        # we do not group the conv filters together
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0), #stride
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2), #or BatchNorm
            nn.MaxPool2d(kernel_size=3, stride=2) # Maxpooling follow LRN
        )
        
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True) #We applied this normalization(LRN) after applying the ReLU nonlinearity in certain layers(first, second)
        )
        
        self.layer4=nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # as well as the fifth convolutional layer(Maxpooling)
        )
        
        self.layer6=nn.Sequential( # First Additional Layer for AlexNet-GAP
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.layer7=nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1000, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1) # No FC because of GAP
        )

    def forward(self, x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out)
        out=self.layer6(out)
        out=self.layer7(out)
        out = out.view(x.size(0), -1)
        
        return out 
    

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0), #stride
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96), #or BatchNorm
            nn.MaxPool2d(kernel_size=3, stride=2) # Maxpooling follow LRN
        )
        
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True) #We applied this normalization(LRN) after applying the ReLU nonlinearity in certain layers(first, second)
        )
        
        self.layer4=nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # as well as the fifth convolutional layer(Maxpooling)
        )
        
        self.fc1=nn.Sequential(
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5)
        )
        
        self.fc2=nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5)
        )

        self.fc3=nn.Sequential(
            nn.Linear(in_features=4096, out_features=1000)
        )
        
    def init_weights(m):
        if type(m) not in [nn.ReLU, nn.LocalResponseNorm, nn.MaxPool2d, 
                            nn.Sequential, nn.Dropout, AlexNet]:
            torch.nn.init.normal_(m.weight, 0, 0.01)
            m.bias.data.fill_(1)
        
    def forward(self, x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out)
        
        out = out.view(out.size(0), -1)

        out=self.fc1(out)
        out=self.fc2(out)
        out=self.fc3(out)

        return out
    
    