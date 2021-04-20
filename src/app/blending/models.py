import torch
import torch.nn as nn
from collections import namedtuple
from torchvision import models

class VGG16_Model(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.style_layers = ['relu1_2', 'relu3_3', 'relu4_3']
        self.content_layers = ['relu2_2']
                
    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h

        vgg_outputs = {
            'relu1_2': h_relu1_2,
            'relu2_2': h_relu2_2,
            'relu3_3': h_relu3_3,
            'relu4_3': h_relu4_3
        }
        return vgg_outputs

class VGG11_Model(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg11(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(5, 10):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(10, 15):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.style_layers = ['relu1_2', 'relu3_3', 'relu4_3']
        self.content_layers = ['relu2_2']

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        
        vgg_outputs = {
            'relu1_2': h_relu1_2,
            'relu2_2': h_relu2_2,
            'relu3_3': h_relu3_3,
            'relu4_3': h_relu4_3
        }
        return vgg_outputs
    
class MeanShift(nn.Conv2d):
    def __init__(self):
        super().__init__(3, 3, kernel_size=1)
        rgb_range=1
        rgb_mean=(0.4488, 0.4371, 0.4040)
        rgb_std=(1.0, 1.0, 1.0)
        sign=-1
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False