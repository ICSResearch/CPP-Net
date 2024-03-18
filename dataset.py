import os
import torch
import torchvision
from torch.utils import data
import numpy as np
from PIL import Image
import scipy.io as scio
import glob
from typing import Any

try:
    import accimage

    torchvision.set_image_backend("accimage")
except:
    pass
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def pil_loader(path: str) -> Image.Image:
    return Image.open(path)


def accimage_loader(path: str) -> Any:
    # import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CustomDataset(data.Dataset):
    def __init__(self, file_path, transforms=None):
        self.images = glob.glob(os.path.join(file_path, "*"))
        if transforms is None:
            self.transforms = torchvision.transforms.ToTensor()
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img = self.transforms(default_loader(self.images[index]))
        label = img
        return img, label

    def __len__(self):
        return len(self.images)


def build_dataset(is_train, args):
    if is_train:
        augs = torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.RandomResizedCrop(
                    (args.input_size, args.input_size)
                ),
                torchvision.transforms.RandomRotation(degrees=45),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.ToTensor(),
            ]
        )
        dataset = CustomDataset(args.data_path, transforms=augs)
    else:
        augs = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((args.input_size, args.input_size)),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ]
        )
        dataset = CustomDataset(args.eval_data_path, transforms=augs)
    return dataset
