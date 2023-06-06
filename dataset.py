import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from collections import defaultdict
import multiprocessing

from classes import IMAGENET2012_CLASSES

class ImageNet(Dataset):
    def __init__(self, type="train"):
        assert type in ["train", "val", "test"], "Choose valid dataset type."

        if type == "train":
            folders = [f"inputs/imagenet-1k/data/train_images_{i}" for i in range(5)]
        else:
            folders = ["inputs/imagenet-1k/data/" + type + "_images"]

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        #self.data = defaultdict()
        #print("Loading", type, "dataset...")
        self.img_list = []
        for folder in folders:
            self.img_list += [os.path.join(folder, item) for item in os.listdir(folder)[:10000]]

        # self.data = defaultdict()
        # for i, img_path in tqdm(enumerate(img_list), total=len(img_list)):
        #     self.data[i] = self.load_sample(img_path)

        #self.data = process_map(self.load_sample, img_list, chunksize=200)

    def __getitem__(self, item):
        return self.load_sample(self.img_list[item])

    def __len__(self):
        return len(self.img_list)

    def load_sample(self, img_path):
        label = os.path.splitext(img_path)[0].split("_")[-1]
        label = np.argmax(np.array(list(IMAGENET2012_CLASSES.keys())) == label)

        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.preprocess(img)

        sample = defaultdict()
        sample["img"] = img
        sample["label"] = label

        return sample

