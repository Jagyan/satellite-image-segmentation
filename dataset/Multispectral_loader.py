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

class SatelliteMultiSpectralDataset(data.Dataset):

    def __init__(self, data_path, multispectral=True):
        self.data_path = data_path
        self.image_ids = os.listdir(os.path.join(data_path, "images"))
        self.multispectral = multispectral
        self.n_class = 3

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img, lab = image_reader(self.data_path, self.image_ids[idx])
        multispectral_image, label = get_multispectral(img, lab)
        multispectral_image = np.transpose(multispectral_image, (0,1,2))
        multispectral_image = np.int16(multispectral_image)
        multispectral_image = np.clip(multispectral_image,0,2000)
        multispectral_image = multispectral_image/2000 * 255
        multispectral_image = np.float32(multispectral_image)
        multispectral_image = torch.from_numpy(multispectral_image)
        label = torch.from_numpy(label)

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1
        
        return {'X':multispectral_image, 'Y':target, 'l':label}

def image_reader(data_path, id):
    image = gdal.Open(os.path.join(data_path, "images", id, "07.tif"))
    label = gdal.Open(os.path.join(data_path, "labels", id, "dlt.tif"))
    return image, label

def get_multispectral(image, lab):
    # fetching all 12 bands
    bands = image.RasterCount
    img = []
    for band in range(1, bands+1):
        band = image.GetRasterBand(band)

        b = band.ReadAsArray()
        img.append(b)

    img = np.stack(img)

    l_band = lab.GetRasterBand(1)
    label = l_band.ReadAsArray()

    return img, label

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
    satellite_images = SatelliteMultiSpectralDataset("/home/mahapatro/Satellite_image_segmentation/dlt_32N_07")
    for i, data in enumerate(satellite_images):
        print(np.unique(data['Y']))

if __name__ == '__main__':
    main()
