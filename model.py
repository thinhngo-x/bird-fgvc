import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import models
from timm import create_model
import pytorch_lightning as pl
import utils
from dataset import get_birds
from torchmetrics import Accuracy


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


class TimmModel(pl.LightningModule):
    def __init__(self, model_name, use_pretrained, args, stage=1) -> None:
        super().__init__()
        self.stage = stage
        self.net = create_model(model_name, use_pretrained, num_classes=NCLASSES)
        # Freeze all layers
        self.args = args
        for param in self.net.parameters():
            param.requires_grad = False
        self.accuracy = Accuracy()
    
    def unfreeze_layers(self, num_layers):
        if num_layers is None:
            for param in self.net.parameters():
                param.requires_grad = True
            return 0
        modules = [m for m in self.net.modules() if not isinstance(m, nn.Sequential)]
        if num_layers == 1:
            modules = [m for m in self.net.get_classifier().modules() if not isinstance(m, nn.Sequential)]
        for layer in modules[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def unfreeze_fc(self):
        for param in self.net.head.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.net(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                # optimizer, step_size=self.args.step_size, gamma=0.6),
                optimizer, T_0=self.args.step_size, T_mult=2, eta_min=self.args.lr/10),
            'interval': 'step'  # called after each training step
        }
        return [optimizer], [lr_scheduler]
    
    def train_dataloader(self):
        train_labeled_set, train_unlabeled_set, _ = get_birds(
            self.args, train_labeled_path='/train_images', train_unlabeled_path='/external',
            val_path='/val_images'
        )
        if self.args.mode == 'supervise':
            train_labeled_loader = torch.utils.data.DataLoader(
                train_labeled_set, batch_size=self.args.batch_size,
                shuffle=True, num_workers=self.args.cpus, drop_last=True
            )
            return train_labeled_loader
        elif self.args.mode == 'fixmatch':
            train_labeled_loader = torch.utils.data.DataLoader(
                train_labeled_set, batch_size=self.args.batch_size,
                shuffle=True, num_workers=self.args.cpus, drop_last=True
            )
            train_unlabeled_loader = torch.utils.data.DataLoader(
                train_unlabeled_set, batch_size=self.args.batch_size*self.args.mu,
                shuffle=True, num_workers=self.args.cpus, drop_last=True
            )
            return [train_labeled_loader, train_unlabeled_loader]
    
    def val_dataloader(self):
        _, _, val_set = get_birds(
            self.args, train_labeled_path='/train_images', train_unlabeled_path='/.',
            val_path='/val_images'
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=self.args.batch_size, shuffle=False, num_workers=1
        )
        return val_loader
    
    def training_step(self, batch, batch_idx):
        if self.args.mode == 'supervise':
            data, target = batch
            output = self(data)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = criterion(output, target)
            self.log("Training loss", loss, on_step=True)
            return loss
        elif self.args.mode == 'fixmatch':
            labeled_batch, unlabeled_batch = batch
            labeled_data, target = labeled_batch
            unlabeled_data, _ = unlabeled_batch
            weak, strong = unlabeled_data
            inp = torch.cat((labeled_data, weak, strong))
            inp = utils.interleave(inp, 2*self.args.mu+1)
            out = self(inp)
            out = utils.de_interleave(out, 2*self.args.mu+1)
            out_labeled = out[:labeled_data.shape[0]]
            out_weak, out_strong = out[labeled_data.shape[0]:].chunk(2)

            criterion = torch.nn.CrossEntropyLoss(reduction='none')

            loss_spv = criterion(out_labeled, target).mean()

            pseudo_probas = torch.softmax(out_weak.detach(), dim=-1)
            max_probs, targets_pseudo = torch.max(pseudo_probas, dim=-1)
            mask = max_probs.ge(self.args.threshold).float()

            loss_unspv = (criterion(out_strong, targets_pseudo) * mask).mean()

            loss = loss_spv + self.args.lambda_u * loss_unspv
            self.log("Training loss", {"loss": loss, "loss_s": loss_spv,
                                       "loss_u": loss_unspv, "mask": mask.mean().item()})
            return loss
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        preds = torch.argmax(output, dim=1).view(-1).to(self.device)
        acc = self.accuracy(preds, target)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss