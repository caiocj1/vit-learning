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

    train_dataset = ImageFolder("inputs/imagenet/train", transform=val_preprocess)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=512,
                                  num_workers=16,
                                  shuffle=True)

    val_dataset = ImageFolder("inputs/imagenet/val", transform=val_preprocess)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=512,
                                num_workers=16,
                                shuffle=False)

    # ------------------ GET MODEL ------------------
    config_path = os.path.join(os.getcwd(), "config.yaml")
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    vit_params = params["ViTParams"]

    vit_model = ViT(**vit_params).to(device)
    # vit_model = vit_b_32().to(device)
    vit_model = nn.DataParallel(vit_model)

    # ------------------ GET TRAINER AND TRAIN ------------------
    trainer = Trainer(vit_model, train_dataloader, device, args.version, val_dataloader=val_dataloader)
    trainer.train()
