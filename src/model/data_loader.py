import random
from random import randint, choice

import argparse
import matplotlib.pyplot as plt
import PIL
import torch
import yaml
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule
import os


class MyDataset(Dataset):
    def __init__(self, type: str):
        """Create a dataset"""
        super().__init__()
        config_dir = 'config.yaml'
        with open(config_dir) as fin:
            self.config = yaml.safe_load(fin)

        self.ori_images = {image_file.stem: image_file
                           for image_file in Path(f"../../data/pic_store/{type}/ori").glob("*.png")}
        self.src_images = {image_file.stem: image_file
                           for image_file in Path(f"../../data/pic_store/{type}/src").glob("*.png")}
        if (type != "tdrive"):
            self.trg_images = {image_file.stem: image_file
                               for image_file in Path(f"../../data/pic_store/{type}/trg").glob("*.png")}
        self.keys = list(self.ori_images.keys())

        self.image_transform = T.Compose([
            T.Grayscale(1),
            T.ToTensor(),
            T.Lambda(self.trans_img),
        ])
        # print(self.ori_images)
        print(f"dataset_size:{len(self.keys)}")

    def __len__(self):
        return len(self.keys)

    def fix_img(self, img):
        return 1 - img

    def add_noise(self, img, cell_noise=False):
        noise_img = img.clone()
        # 噪声基站，在图片中选择一些位置设为噪声
        if (cell_noise == True):
            noise_x = torch.randint(low=0, high=223, size=(self.config["noise_number"],))
            noise_y = torch.randint(low=0, high=223, size=(self.config["noise_number"],))
            for i in range(self.config["noise_number"]):
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if (0 <= noise_x[i] + x <= 223 and 0 <= noise_y[i] + y <= 223):
                            noise_img[0, noise_x[i], noise_y[i]] = torch.tensor(1.0)

        # 图片整体噪声
        for i in range(img.shape[1]):
            for j in range(img.shape[2]):
                rdn = random.random()
                if rdn < self.config["noise_threshold"]:
                    noise_img[0, i, j] = random.random()

        return noise_img

    def trans_img(self, img):
        # print(img.dtype)
        # 可以用0.5做分界线
        mask = torch.where(img < 0.1, torch.tensor(self.config["mask_value"][1]), img)
        mask = torch.where(img > 0.5, torch.tensor(self.config["mask_value"][3]), mask)
        mask = torch.where((0.1 <= img) * (img <= 0.5), torch.tensor(self.config["mask_value"][2]), mask)

        middle_pos = torch.argwhere(mask == self.config["mask_value"][1])  # 取边界下标
        top, left, down, right = middle_pos[0][1], middle_pos[0][2], \
                                 middle_pos[-1][1], middle_pos[-1][2]
        # 把边框设定为 self.config["mask_value"][0]
        mask[:, :top, :] = torch.tensor(self.config["mask_value"][0])
        mask[:, down + 1:, :] = torch.tensor(self.config["mask_value"][0])
        mask[:, :, :left] = torch.tensor(self.config["mask_value"][0])
        mask[:, :, right + 1:] = torch.tensor(self.config["mask_value"][0])

        img[:, :top, :] = torch.tensor(0.0)
        img[:, down + 1:, :] = torch.tensor(0.0)
        img[:, :, :left] = torch.tensor(0.0)
        img[:, :, right + 1:] = torch.tensor(0.0)

        # visualize
        # plt.figure(figsize=(2, 2))
        # plt.imshow(mask[0].data.numpy())
        # plt.axis('off')
        # plt.show()
        # plt.clf()
        # plt.close('all')

        return img, mask

    def __getitem__(self, ind):
        key = self.keys[ind]
        ori_image = PIL.Image.open(self.ori_images[key])
        src_image = PIL.Image.open(self.src_images[key])
        ori_image_tensor, _ = self.image_transform(ori_image)  # C*H*W
        # ori_image_noise_tensor = self.add_noise(ori_image_tensor)
        src_image_tensor, _ = self.image_transform(src_image)  # C*H*W
        # src_image_noise_tensor = self.add_noise(src_image_tensor,cell_noise=True)
        if (self.config["type"] == "tdrive"):
            trg_image_tensor = torch.tensor([1])
            mask = torch.tensor([1])
        else:
            trg_image = PIL.Image.open(self.trg_images[key])
            trg_image_tensor, mask = self.image_transform(trg_image)  # C*H*W

        # visualize
        # plt.figure(figsize=(2, 2))
        # plt.imshow(ori_image_tensor[0].data.numpy())
        # plt.show()
        # plt.imshow(src_image_tensor[0].data.numpy())
        # plt.show()
        # plt.imshow(trg_image_tensor[0].data.numpy())
        # plt.axis('off')
        # plt.show()
        # plt.clf()
        # plt.close('all')

        input_image_tensor = torch.cat([ori_image_tensor, src_image_tensor], dim=0)  # (2*C)*H*W
        # noise_input_tensor = torch.cat([ori_image_noise_tensor, src_image_noise_tensor], dim=0)  # (2*C)*H*W
        noise_input_tensor = torch.tensor([1])

        return input_image_tensor, trg_image_tensor, mask, noise_input_tensor


class PLDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: int = 128,
                 num_workers=8,
                 shuffle=True,
                 pin_memory=True
                 ):
        """
        Args:
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.train_dataset = MyDataset(type="train")
        self.valid_dataset = MyDataset(type="val")
        self.tdrive_dataset = MyDataset(type="tdrive")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          drop_last=False, collate_fn=self.dl_collate_fn,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          drop_last=False, collate_fn=self.dl_collate_fn,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False, collate_fn=self.dl_collate_fn,
                          pin_memory=self.pin_memory)
    # def test_dataloader(self):
    #     return DataLoader(self.tdrive_dataset, batch_size=self.batch_size, shuffle=False,
    #                       num_workers=self.num_workers,
    #                       drop_last=False, collate_fn=self.dl_collate_fn,
    #                       pin_memory=self.pin_memory)

    def dl_collate_fn(self, batch):
        return torch.stack([row[0] for row in batch], dim=0), \
               torch.stack([row[1] for row in batch], dim=0), \
               torch.stack([row[2] for row in batch], dim=0), \
               torch.stack([row[3] for row in batch], dim=0),
