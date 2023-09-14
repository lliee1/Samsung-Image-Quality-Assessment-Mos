import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from torchvision import transforms

from PIL import Image

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]

ImageFile.LOAD_TRUNCATED_IMAGES = True


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I.convert("RGB")


def get_default_img_loader():
    return functools.partial(image_loader)


def normalize_value(x):
    min_value = 0
    max_value = 10
    new_min = 1
    new_max = 5
    return (x - min_value) / (max_value - min_value) * (new_max - new_min) + new_min


class liqe_dataset(Dataset):
    def __init__(
        self,
        csv_file,
        preprocess,
        num_patch,
        test,
        get_loader=get_default_img_loader,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        print("%d csv data successfully loaded!" % self.__len__())
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test

    def __getitem__(self, index):
        image_name = "/root/dacon/data" + self.data.img_path[index][1:]
        I = self.loader(image_name)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32
        patches = (
            I.unfold(2, kernel_h, step)
            .unfold(3, kernel_w, step)
            .permute(2, 3, 0, 1, 4, 5)
            .reshape(-1, n_channels, kernel_h, kernel_w)
        )
        assert patches.size(0) >= self.num_patch
        # self.num_patch = np.minimum(patches.size(0), self.num_patch)
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
            img_name = self.df.img_name[index]
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch,))
            mos = self.data.mos[index]
            mos = normalize_value(mos)
        patches = patches[sel, ...]

        if self.test:
            sample = {"I": patches, "img_name": img_name}
        else:
            sample = {
                "I": patches,
                "mos": float(mos),
            }

        return sample

    def __len__(self):
        return len(self.data.index)


class AdaptiveResize(object):
    """Resize the input PIL Image to the given size adaptively.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size

        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return transforms.Resize(self.image_size, self.interpolation)(img)

        if h < 384 or w < 384:
            return transforms.Resize((384,384), self.interpolation)(img)
        elif h < self.size or w < self.size:
            return img
        else:
            return transforms.Resize(self.size, self.interpolation)(img)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _preprocess2():
    return Compose(
        [
            _convert_image_to_rgb,
            AdaptiveResize(768),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()
