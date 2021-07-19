import os
import sys
import io
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import models.fcn as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.Multispectral_loader import SatelliteMultiSpectralDataset as Dataset
from modules.trainer import *
from models.fcn_multispectral import VGGNet, FCN32s, FCN16s, FCN8s, FCNs

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mahapatro/Satellite_image_segmentation/dlt_32N_07', help='Dataset directory')
parser.add_argument('--log-dir', default='outputs', help='Log dir')
parser.add_argument('--batch-size', type=int, default=8, help='Batch Size during training [default: 4]')
parser.add_argument('--num-epochs', type=int, default=50, help='Epochs to run [default: 200000]')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate [default: 1e-5]')
parser.add_argument('--use-pretrained', type=bool, default=True, help='Use pretrained model')
parser.add_argument('--checkpoint-dir', default='/home/mahapatro/Satellite_image_segmentation/outputs', help='Path to model')

args = parser.parse_args()

checkpoint_dir = os.path.join(args.checkpoint_dir, 'checkpoints_multispectral')

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir)

Satellite_Dataset = Dataset(args.data_dir)

dataset_len = len(Satellite_Dataset)

train_count = int(0.8 * dataset_len)
val_count = dataset_len - train_count

print('Building dataloaders')
train_dataset, valid_dataset = torch.utils.data.random_split(Satellite_Dataset, (train_count, val_count))

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True, batch_size=args.batch_size, drop_last = True)
val_dataloader = torch.utils.data.DataLoader(valid_dataset, shuffle = True, batch_size=args.batch_size, drop_last = True)

writer = SummaryWriter()

configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(args.batch_size, args.num_epochs, 50, 0.5, args.learning_rate, 0, 1e-5)
print("Configs:", configs)

# create dir for score
score_dir = os.path.join(args.log_dir, "scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores    = np.zeros((args.num_epochs, 3))
pixel_scores = np.zeros(args.num_epochs)

vgg_model = VGGNet(pretrained=False, requires_grad=True, remove_fc=True)
fcn_model = FCNs(pretrained_net=vgg_model, n_class=3)

ts = time.time()
vgg_model = vgg_model.cuda()
fcn_model = fcn_model.cuda()
fcn_model = nn.DataParallel(fcn_model)
if(args.use_pretrained):
    print("Using pretrained model")
    checkpoint = torch.load(os.path.join(checkpoint_path, "49_checkpoint.pth.tar"))
    fcn_model = checkpoint
print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

my_train(fcn_model, args.num_epochs, train_loader, val_dataloader, args.learning_rate, writer, checkpoint_path, IU_scores, pixel_scores, score_dir)