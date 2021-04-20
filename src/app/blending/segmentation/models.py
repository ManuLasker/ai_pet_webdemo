from numpy.core.fromnumeric import prod
import pytorch_lightning as pl
import torch
import numpy as np
import torch.optim as optim
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)
    
class VG11BackBoneSegmentation(nn.Module):
    def __init__(self, requires_grad:bool = False):
        super().__init__()
        vgg16 = models.vgg11(pretrained=True)
        self.vgg_pretrained_features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.classifier = vgg16.classifier
        if not requires_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False
        self.out_features = np.prod(self(torch.zeros(1, 3, 224, 224)).shape)
    
    def forward(self, x):
        x = self.vgg_pretrained_features(x)
        x = self.avgpool(x)
        return x
    
class VG16BackBoneSegmentation(nn.Module):
    def __init__(self, requires_grad:bool = False):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.vgg_pretrained_features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.classifier = vgg16.classifier
        if not requires_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False
        self.out_features = np.prod(self(torch.zeros(1, 3, 224, 224)).shape)


    def forward(self, x):
        x = self.vgg_pretrained_features(x)
        x = self.avgpool(x)
        return x

class BackboneNaive(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 3, 25, 25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 3)
        )
        self.out_features = np.prod(self(torch.zeros(1, 3, 224, 224)).shape)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class SegmentationNaive(nn.Module):
    def __init__(self, img_shape=(224, 224),
                 requires_grad_backbone:bool = False,
                 backbone:int = 11):
        super().__init__()
        if backbone == 11:
            self.backbone = VG11BackBoneSegmentation(requires_grad=requires_grad_backbone)
        elif backbone == 16:
            self.backbone = VG16BackBoneSegmentation(requires_grad=requires_grad_backbone)
        elif backbone == 0:
            self.backbone = BackboneNaive()
            
        self.decoder = nn.Sequential(
            Lambda(reshape_batch),
            nn.Linear(in_features=self.backbone.out_features,
                      out_features=np.prod(img_shape)),
        )
    
    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.decoder(x)
        return x

def reshape_batch(x:torch.Tensor):
    return x.view(x.size(0), -1)

class VGGFCN(nn.Module):
    def __init__(self, backbone:int = 16, requires_grad_backbone:bool = False):
        super().__init__()
        vgg = VG16BackBoneSegmentation(requires_grad_backbone)
        self.vgg_features = vgg.vgg_pretrained_features
        self.vgg_avgpool = vgg.avgpool
        vgg_classifier = vgg.classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(4096, 1000, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(1000, 1, kernel_size=1)
        )
        # Process to transfer weights from linear layers
        # into convolutionar layers
        self._initialize_weights(vgg_classifier)
        for param in self.classifier.parameters():
            param.requires_grad = True
        
    def _initialize_weights(self, vgg_classifier):
        self.classifier[0].weight.data = (
            vgg_classifier[0].weight.data.view(
                self.classifier[0].weight.size()
            )
        )
        self.classifier[3].weight.data = (
            vgg_classifier[3].weight.data.view(
                self.classifier[3].weight.size()
            )
        )
        self.classifier[6].weight.data = (
            vgg_classifier[6].weight.data.view(
                self.classifier[6].weight.size()
            )
        )
        
    def forward(self, x:torch.Tensor):
        bs, _, h, w = x.size()
        x = self.vgg_features(x)
        x = self.vgg_avgpool(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(h, w),
                         mode='bilinear', align_corners=True)
        return x
    
class SegmentationModule(pl.LightningModule):
    def __init__(self, model:nn.Module, loss,
                 is_fcn:bool=True):
        super().__init__()
        self.model = model
        self.is_fcn = is_fcn
        self.loss = loss
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        batch = train_batch
        x, y = batch['source'], batch['mask']
        bs, _, _, _ = x.size()
        y_pred = self(x)
        if not self.is_fcn:
            loss = self.loss(y_pred, y.view(bs, -1))
        else:
            loss = self.loss(y_pred, y)
        self.log('train_loss', loss, prog_bar=True, logger=False,
                on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        batch = val_batch
        x, y = batch['source'], batch['mask']
        bs, _, _, _ = x.size()
        y_pred = self(x)
        if not self.is_fcn:
            loss = self.loss(y_pred, y.view(bs, -1))
        else:
            loss = self.loss(y_pred, y)
        self.log('val_loss', loss, prog_bar=True, logger=False,
                on_step=False, on_epoch=True)
    
class Predictor:
    model = None
    model_path = None
    
    @classmethod
    def set_config(cls, model_path:str):
        cls.model_path = model_path
        
    @classmethod
    def load_model(cls, model_path:str):
        if cls.model is None:
            cls.model = torch.jit.load(model_path)
            # cls.model = torch.load(model_path)
        return cls.model
    
    @classmethod
    def predict(cls, x: torch.Tensor):
        model = cls.load_model(cls.model_path)
        model.eval()
        with torch.no_grad():
            prediction: torch.Tensor = model(x)
        # return prediction.softmax(dim=1).argmax(dim=1)
        return prediction
    
    @classmethod
    def export_torch_script(cls, model_exported_path):
        x: torch.Tensor = torch.rand(1, 3, 224, 224)
        model = cls.load_model(cls.model_path)
        model.eval()
        traced_script_module = torch.jit.trace(model, x)
        traced_script_module.save(model_exported_path)