import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchinfo import summary

NCLASSES = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, NCLASSES)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ResNet(nn.Module):
    def __init__(self, netName, pretrained) -> None:
        """Initialize a resnet model.
        
        Args:
            netName (String): resnet18, resnet34, resnet50, resnet101, resnet152
            preTrained (Boolean)
        """
        super(ResNet, self).__init__()
        self.net = getattr(models, netName)(pretrained=pretrained)
        for param in self.net.parameters():
            param.requires_grad = False
        for param in self.net.layer4.parameters():
            param.requires_grad = True
        num_feats_extr = self.net.fc.in_features
        self.net.fc = nn.Sequential(
            nn.Linear(num_feats_extr, NCLASSES),
        )
    
    def forward(self, x):
        x = self.net(x)
        return x
    
    def extract_feats(self, x):
        modules = list(self.net.children())[:-1]
        extrtor = nn.Sequential(*modules)
        return extrtor(x)


class MaskVC(nn.Module):
    def __init__(self) -> None:
        super(MaskVC, self).__init__()
        self.maskrcnn = models.detection.maskrcnn_resnet50_fpn(pretrained=True, num_classes=1)
        for param in self.net.parameters():
            param.requires_grad = False
        # in_features = self.maskrcnn.roi_heads.box_predictor.cls_score.in_features
        # self.maskrcnn.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        #     in_features, 2
        # )
        # in_features_mask = self.maskrcnn.roi_heads.mask_predictor.conv5_mask.in_channels
        # hidden_layer = 256
        # self.maskrcnn.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
        #     in_features_mask, hidden_layer, 2
        # )

        self.resnet = ResNet('resnext101_32x8d', pretrained=True)
    
    def forward(self, x):
        mask_preds = self.maskrcnn(x)['masks']
        print(mask_preds)
        return x



class EfficientNet(nn.Module):
    def __init__(self, netName, pretrained) -> None:
        """Initialize a resnet model.
        
        Args:
            netName (String): resnet18, resnet34, resnet50, resnet101, resnet152
            preTrained (Boolean)
        """
        super(EfficientNet, self).__init__()
        self.net = getattr(models, netName)(pretrained=pretrained)
        for param in self.net.parameters():
            param.requires_grad = False
        num_feats_extr = self.net.classifier[1].in_features
        self.net.classifier = nn.Sequential(
            nn.Linear(num_feats_extr, NCLASSES)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, netName, pretrained) -> None:
        """Initialize a resnet model.
        
        Args:
            netName (String): densenet121, densenet161, densenet169, densenet201
            preTrained (Boolean)
        """
        super(DenseNet, self).__init__()
        self.net = getattr(models, netName)(pretrained=pretrained)
        for param in self.net.parameters():
            param.requires_grad = False
        num_feats_extr = self.net.classifier.in_features
        self.net.classifier = nn.Sequential(
            nn.Linear(num_feats_extr, NCLASSES),
        )
    
    def forward(self, x):
        x = self.net(x)
        return x