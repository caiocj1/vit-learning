import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import vit_b_32
from torchvision import transforms
import os

from trainer import Trainer
from models.vit import ViT

# ---------------------------------------
from pathlib import Path
import json
from functools import wraps

def file_cache(filename):
    """Decorator to cache the output of a function to disk."""
    def decorator(f):
        @wraps(f)
        def decorated(self, directory, *args, **kwargs):
            filepath = Path(directory) / filename
            if filepath.is_file():
                out = json.loads(filepath.read_text())
            else:
                out = f(self, directory, *args, **kwargs)
                filepath.write_text(json.dumps(out))
            return out
        return decorated
    return decorator

class CachedImageFolder(ImageFolder):
    @file_cache(filename="cached_classes.json")
    def find_classes(self, directory, *args, **kwargs):
        classes = super().find_classes(directory, *args, **kwargs)
        return classes

    @file_cache(filename="cached_structure.json")
    def make_dataset(self, directory, *args, **kwargs):
        dataset = super().make_dataset(directory, *args, **kwargs)
        return dataset

# ---------------------------------------

if __name__ == "__main__":
    # ------------------ ARGUMENT PARSING ------------------
    parser = argparse.ArgumentParser(description="ViT training", allow_abbrev=False)

    parser.add_argument("--version", "-v", required=True, type=str, help="Version name for TensorBoard.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ LOAD DATA ------------------
    val_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_preprocess = transforms.Compose([
        transforms.RandAugment(num_ops=2, magnitude=15),
        val_preprocess
    ])

    train_dataset = CachedImageFolder("inputs/imagenet/train", transform=val_preprocess)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=1024,
                                  num_workers=16,
                                  shuffle=True)

    val_dataset = CachedImageFolder("inputs/imagenet/val", transform=val_preprocess)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1024,
                                num_workers=16,
                                shuffle=False)

    # ------------------ GET MODEL ------------------
    # config_path = os.path.join(os.getcwd(), "config.yaml")
    # with open(config_path) as f:
    #     params = yaml.load(f, Loader=yaml.SafeLoader)
    # vit_params = params["ViTParams"]
    #
    # vit_model = ViT(**vit_params).to(device)
    # vit_model = nn.DataParallel(vit_model)

    vit_model = vit_b_32().to(device)

    # ------------------ GET TRAINER AND TRAIN ------------------
    trainer = Trainer(vit_model, train_dataloader, device, args.version, val_dataloader=val_dataloader)
    trainer.train()
