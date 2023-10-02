import torch.nn as nn

class MultiLayerCNN(nn.Module):
    def __init__(self, num_class, features=[16,'M',16,16], drop=0.):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(16*16*16,num_class)
        )
        self.layers = []
        
        in_channel=3
        for i in features:
            if i == 'M':
                self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                self.layers += [
                    nn.Conv2d(in_channel, i, kernel_size=3, padding=1),
                    nn.BatchNorm2d(i),
                    nn.ReLU(inplace=True)
                ]
                in_channel = i
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x