import torch
import torch.nn as nn
import torch.nn.functional as F

class VggNet(nn.Module):

    def __init__(self, features):
        # features : [64, 64, M, ...]
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*4*4, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10), # num class = 10 (CIFAR10)
        )

        self.layers = [] # construct layers using 'features' paremeter
        in_channels = 4
        for i in features:
            if i == 'M': #MaxPooling layer
                self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else: #Convolution + Batchnorm + Relu
                conv2d = nn.Conv2d(in_channels, i, kernel_size=3, padding=1)
                self.layers += [conv2d, nn.BatchNorm2d(i), nn.ReLU(inplace=True)]
                in_channels = i
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1) #flattens the tensor with batch size
        x = self.classifier(x)
        return x

