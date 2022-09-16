import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import natsort

import cv2
from torchvision.transforms import transforms


class sonarDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, transform=None):
        # dataset path
        self.root = root
        self.CLASSES = classes
        self.transform = transform

        self.imgs = list(natsort.natsorted(os.listdir(os.path.join(root, "Images"))))
        self.masks = list(natsort.natsorted(os.listdir(os.path.join(root, "Masks"))))

        self.class_values = [i for i in range(0,len(self.CLASSES))]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images ad masks
        img = cv2.imread(os.path.join(self.root, "Images", self.imgs[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.root, "Masks", self.masks[idx]), cv2.IMREAD_GRAYSCALE)

        img = np.array(img) / 255.
        img = np.expand_dims(img, axis=0).astype(np.float32)

        mask = torch.from_numpy(mask).long()

        sample = {'image': img, 'mask':mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # standard scaling would be probably better then dividing by 255 (subtract mean and divide by std of the dataset)
        image = np.array(image) / 255.

        print("To tensor")
        sample = {'image': torch.from_numpy(image).permute(2, 0, 1).float(),
                  'mask': torch.from_numpy(mask).long(),
                  }

        return sample