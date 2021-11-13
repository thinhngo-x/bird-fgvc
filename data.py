import zipfile
import os

import torchvision.transforms as transforms


train_data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation((-30, 30)),
    transforms.RandomCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


