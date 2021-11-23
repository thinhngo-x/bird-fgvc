import argparse
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer


# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=16, metavar='B',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--step-size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--lambda-u', default=1.0, type=float,
                    help='coefficient of unlabeled loss')
parser.add_argument('--threshold', default=0.95, type=float,
                    help='pseudo label threshold')
parser.add_argument('--mu', default=2, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--mode', default='fixmatch', type=str, choices=['fixmatch', 'supervise'])
parser.add_argument('--gpus', default=1, type=int)
parser.add_argument('--num-nodes', default=1, type=int)
parser.add_argument('--cpus', default=12, type=int)
parser.add_argument('--stage', type=int)
parser.add_argument('--model', type=str)

parser.add_argument('--use-pretrained', dest='use_pretrained', action='store_true')
parser.add_argument('--from-scratch', dest='use_pretrained', action='store_false')
parser.set_defaults(use_pretrained=True)
args = parser.parse_args()


from model import TimmModel
model = TimmModel(args.model, args.use_pretrained, args)
early_stop_cb = EarlyStopping(monitor="val_loss", patience=5)
ckpt_cb = ModelCheckpoint(monitor="val_loss", dirpath="ckpts", filename=args.experiment+'_'+args.model)
lr_monitor = LearningRateMonitor(logging_interval=None)

if args.stage == 1:
    model.unfreeze_layers(1)
    trainer = Trainer(
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        strategy='ddp',
        max_epochs=args.epochs,
        progress_bar_refresh_rate=20,
        callbacks=[lr_monitor, early_stop_cb, ckpt_cb],
        log_every_n_steps=5
    )
    trainer.fit(model)

elif args.stage == 2:

    model = TimmModel.load_from_checkpoint("ckpts/s1_sp_coslr_1e-1_step_50_decay_01_autoaug_"+args.model+".ckpt", model_name=args.model,
                                           use_pretrained=False, args=args, stage=args.stage)
    model.unfreeze_layers(120)
    trainer = Trainer(
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        strategy='ddp',
        max_epochs=args.epochs,
        log_every_n_steps=5,
        callbacks=[early_stop_cb, ckpt_cb]
    )
    trainer.fit(model)