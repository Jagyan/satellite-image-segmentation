from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile as tiff
from PIL import Image
import cv2
import pandas as pd

import torch
import torch.utils.data as data
from torchvision import utils

class SatelliteDataset(data.Dataset):

    def __init__(self, data_path, multispectral=True):
        self.data_path = data_path
        self.image_ids = os.listdir(os.path.join(data_path, "images"))
        self.multispectral = multispectral
        self.n_class = 3

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img, lab = image_reader(self.data_path, self.image_ids[idx])
        rgb_image, label = get_rgb(img, lab)
        rgb_image = np.transpose(rgb_image[:,:,[2,1,0]])
        rgb_image = np.int16(rgb_image)
        rgb_image = np.clip(rgb_image,0,2000)
        rgb_image = rgb_image/2000 * 255
        rgb_image = np.float32(rgb_image)
        rgb_image = torch.from_numpy(rgb_image)
        label = torch.from_numpy(label)

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        return {'X':rgb_image, 'Y':target, 'l':label}

def image_reader(data_path, id):
    image = gdal.Open(os.path.join(data_path, "images", id, "07.tif"))
    label = gdal.Open(os.path.join(data_path, "labels", id, "dlt.tif"))
    return image, label

def get_rgb(image, lab):
    # since there are 3 bands
    # we store in 3 different variables
    band1 = image.GetRasterBand(3) # Red channel
    band2 = image.GetRasterBand(2) # Green channel
    band3 = image.GetRasterBand(1) # Blue channel

    b1 = band1.ReadAsArray()
    b2 = band2.ReadAsArray()
    b3 = band3.ReadAsArray()

    img = np.dstack((b1, b2, b3))
    # f = plt.figure()
    # plt.imshow(img)
    # plt.show()

    l_band = lab.GetRasterBand(1)
    label = l_band.ReadAsArray()

    return img, label

    # f1 = plt.figure()
    # plt.imshow(l)
    # plt.show()

def displayfile(data_path, id):
    img = tiff.imread(os.path.join(data_path, "images", id, "07.tif"))
    rgb_image = img[:,:,[3, 2, 1]]
    rgb_image = np.int16(rgb_image)
    rgb_image = np.clip(rgb_image,0,1500)
    rgb_image = rgb_image/1500 * 255
    rgb_image = np.float32(rgb_image)
    cv2.imshow("image", rgb_image)
    cv2.waitKey()

def main():
    # image, label = image_reader("C:/Users/yagya/OneDrive/Desktop/Vision_impulse/dlt_32N_07", "32N-12E-230N_03_23")
    # plot_image(image, label)
    satellite_images = SatelliteDataset("/home/mahapatro/Satellite_image_segmentation/dlt_32N_07")
    for i, data in enumerate(satellite_images):
        print(np.unique(data['Y']))

if __name__ == '__main__':
    main()
