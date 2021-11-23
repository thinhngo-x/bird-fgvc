import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np


IMG_SIZE = 384
TEST_RESIZE = 448
CROP_PADDING = int(IMG_SIZE * 0.125)
NUM_AUG_OPS = 2
MAGNITUDE = 10
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


data_transforms = transforms.Compose([
    transforms.Resize(TEST_RESIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


class TransformFixMatch(object):
    def __init__(self, mean, std) -> None:
        self.weak = transforms.Compose([
            transforms.Resize(TEST_RESIZE),
            transforms.RandomCrop(IMG_SIZE),
            # transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip()
        ])
        self.strong = transforms.Compose([
            transforms.Resize(TEST_RESIZE),
            transforms.RandomCrop(IMG_SIZE),
            # transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(NUM_AUG_OPS, MAGNITUDE)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class BirdDataset(datasets.ImageFolder):
    def __init__(self, root, idxs, transform):
        super().__init__(root, transform=transform)
        if idxs is not None:
            self.samples = np.array(self.samples)[idxs]
            self.targets = np.array(self.targets)[idxs]
    
    def __getitem__(self, index: int):
        return super().__getitem__(index)


def get_birds(args, train_labeled_path, train_unlabeled_path, val_path):
    transforms_labeled = transforms.Compose([
        transforms.Resize(TEST_RESIZE),
        transforms.RandomCrop(IMG_SIZE),
        # transforms.RandomResizedCrop(IMG_SIZE,scale=(0.2,1.0),ratio=(0.9,1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    train_labeled_dataset = datasets.ImageFolder(
        args.data + train_labeled_path, transform=transforms_labeled
    )

    train_unlabeled_dataset = datasets.ImageFolder(args.data + train_unlabeled_path)

    idxs = resample_idxs(train_unlabeled_dataset.targets, len(train_labeled_dataset.targets)*args.mu)
    assert len(idxs) == len(train_labeled_dataset.targets)*args.mu
    train_unlabeled_dataset = BirdDataset(
        args.data + train_unlabeled_path, idxs,
        transform=TransformFixMatch(MEAN, STD)
    )

    val_dataset = datasets.ImageFolder(
        args.data + val_path, transform=data_transforms
    )

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset


def resample_idxs(labels, num_labels_final):
    diff = num_labels_final - len(labels)
    idxs = np.array(range(len(labels)))
    if diff > 0:
        expand = np.random.choice(idxs, diff, replace=True)
        return np.concatenate((idxs, expand))
    else:
        return np.random.choice(idxs, num_labels_final, replace=False)
