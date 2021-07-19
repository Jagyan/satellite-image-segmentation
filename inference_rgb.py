import os
import sys
import io
import random
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

from dataset.Satellite_loader import *
from modules.trainer import *
from models.fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/home/mahapatro/Satellite_image_segmentation/Test_data', help='Dataset directory')
    parser.add_argument('--results-dir', default='results', help='Results dir')
    parser.add_argument('--test-count', type=int, default=10, help='Number of test samples')
    parser.add_argument('--checkpoint-dir', default='/home/mahapatro/Satellite_image_segmentation/outputs', help='Path to model')

    args = parser.parse_args()

    # Setting result directory
    results_dir = os.path.join(args.data_dir, args.results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Fetching model checkpoint
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'checkpoints')
    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=3)
    fcn_model = nn.DataParallel(fcn_model)
    checkpoint = torch.load(os.path.join(checkpoint_dir, "0_checkpoint.pth.tar"))
    fcn_model = checkpoint

    # Fetching sample image ids
    image_ids = os.listdir(os.path.join(args.data_dir, "images"))
    sample_ids = random.sample(image_ids, args.test_count)

    # Fetching sample images
    for id in sample_ids:
        img, lab = image_reader(args.data_dir, id)
        rgb_image, label = get_rgb(img, lab)
        rgb_image = np.transpose(rgb_image[:,:,[2,1,0]])
        rgb_image = np.int16(rgb_image)
        rgb_image = np.clip(rgb_image,0,2000)
        rgb_image = rgb_image/2000 * 255
        rgb_image = np.float32(rgb_image)
        rgb_image = np.expand_dims(rgb_image, 0)
        label = np.expand_dims(label, 0)
        test_result = test(fcn_model, torch.tensor(rgb_image), torch.tensor(label), id)
        save_result(label, test_result, id, results_dir)

# Running the image through the model
def test(model, image, label, id):
    total_ious = []
    pixel_accs = []

    output = model(image)
    output = output.data.cpu().numpy()
        
    N, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, 3).argmax(axis=1).reshape(N, h, w)
    target = label.reshape(N, h, w)
    for p, t in zip(pred, target):
        total_ious.append(iou(p, t))

    # Quantitative analysis of the result
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    print("Image id{}, meanIoU: {}, IoUs: {}".format(id, np.nanmean(ious), ious))

    return pred

# Save the output to the result path
def save_result(target, prediction, id, results_dir):
    sample_path = os.path.join(results_dir, id)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    f1 = plt.figure()
    plt.imsave(os.path.join(sample_path, "target.png"), target[0])
    
    plt.imsave(os.path.join(sample_path, "prediction.png"), prediction[0])

if __name__ == '__main__':
    main()