# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os

def my_train(model, num_epochs, train_loader, val_loader, learning_rate, summaryWriter, checkpoint_path, IUscores, pixel_scores, score_dir):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for i in range(num_epochs):
        scheduler.step()
        ts = time.time()
        train(model, train_loader, summaryWriter, optimizer, i, num_epochs, criterion)
        print("Finish epoch {}, time elapsed {}".format(i, time.time() - ts))
        val(model, val_loader, summaryWriter, optimizer, i, num_epochs, criterion, IUscores, pixel_scores, score_dir)
        filename = str(i) + '_checkpoint.pth.tar'
        torch.save(model, os.path.join(checkpoint_path, filename))

def train(model, train_loader, summaryWriter, optim, epoch, num_epochs, criterion):
    model.train()
    for i_batch, data in enumerate(train_loader):
        optim.zero_grad()

        input = Variable(data['X'].cuda())
        label = Variable(data['Y'].cuda())

        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optim.step()

        if (i_batch % 20) == 0:
            step = epoch * len(train_loader) * i_batch
            summaryWriter.add_scalar("Training_Loss", loss, step)
            print("epoch{}, iter{}, loss: {}".format(epoch, i_batch, loss))


def val(model, train_loader, summaryWriter, optim, epoch, num_epochs, criterion, IU_scores, pixel_scores, score_dir):
    model.eval()
    total_ious = []
    pixel_accs = []
    for i_batch, data in enumerate(train_loader):
        optim.zero_grad()

        input = Variable(data['X'].cuda())
        label = Variable(data['Y'].cuda())

        output = model(input)
        output = output.data.cpu().numpy()
        
        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, 3).argmax(axis=1).reshape(N, h, w)

        target = data['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)
    summaryWriter.add_scalar("Validation_IOU", np.nanmean(ious), epoch)
    summaryWriter.add_scalar("Validation_Pixelacc", pixel_accs, epoch)

# Function for evaluation of result using Intersection over union
def iou(pred, target):
    ious = []
    for cls in range(3):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious

# Function for evaluation of result using Pixel wise accuracy
def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total