import argparse
from inspect import getblock
import os
import torch
from torch import random
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import torch
import utils
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sn
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import data_transforms, train_data_transforms

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=train_data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import DenseNet, Net, ResNet, EfficientNet
model = ResNet('resnext101_32x8d', pretrained=True)
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.01)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    writer.add_scalar("Loss/train", loss.data.item(), epoch)
    writer.flush()


def validation(epoch, num_embeds=None, confusion_mat=True):
    model.eval()
    validation_loss = 0
    correct = 0
    if num_embeds is not None:
        embeddings = []
        cls_labels = []
        imgs = []
    if confusion_mat:
        preds = []
        targets = []
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        if num_embeds is not None:
            embeddings.append(model.extract_feats(data).detach().cpu())
            cls_labels += [*target.detach().cpu().view(-1)]
            imgs.append(data.detach().cpu())
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        if confusion_mat:
            targets += [*target.view(-1).detach().cpu()]
            preds += [*pred.view(-1).detach().cpu()]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    writer.add_scalar("Loss/val", validation_loss, epoch)
    writer.add_scalar("Eval/val", 100. * correct / len(val_loader.dataset), epoch)
    if num_embeds is not None:
        embeddings = torch.cat(embeddings)
        embeddings = embeddings.view(embeddings.shape[0], -1)
        imgs = torch.cat(imgs)
        imgs.view(-1, 224, 224)
        writer.add_embedding(
            embeddings,
            metadata=cls_labels,
            label_img=imgs,
            global_step=epoch
        )
    if confusion_mat:
        conf_mat = confusion_matrix(preds, targets)
        df_cm = pd.DataFrame(conf_mat, range(20), range(20))
        sn.set(font_scale=1.4)
        fig = sn.heatmap(df_cm, annot=True, annot_kws={"size":16}).get_figure()
        writer.add_figure('confusion matrix', fig, global_step=epoch)
        
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return 100. * correct.item() / len(val_loader.dataset)


for epoch in range(1, args.epochs + 1):
    train(epoch)
    val_score = validation(epoch, num_embeds=None, confusion_mat=True)
    model_file = args.experiment + '/model_' + str(epoch) + '.pth'
    utils.save_ckpt_auto_rm(model, model_file, val_score, max_num=5)
    print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
