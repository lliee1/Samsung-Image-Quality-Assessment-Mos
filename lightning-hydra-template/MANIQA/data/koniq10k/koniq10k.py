import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image


class Koniq10k(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(Koniq10k, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        with open(self.txt_file_name, "r") as listFile:
            for line in listFile:
                dis, score = line.split()
                if dis in list_name:
                    score = float(score)
                    dis_files_data.append(dis)
                    score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = list(score_data.astype("float").reshape(-1, 1))

        self.data_dict = {"d_img_list": dis_files_data, "score_list": score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict["d_img_list"])

    def __getitem__(self, idx):
        d_img_name = self.data_dict["d_img_list"][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.data_dict["score_list"][idx]

        sample = {"d_img_org": d_img, "score": score}
        if self.transform:
            sample = self.transform(sample)
        return sample


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample


class MyDataset_with_blur(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)

        # apply blur or not
        p_aug = np.array([0.5, 0.5])
        prob_lr = np.random.choice([1, 0], p=p_aug.ravel())
        if prob_lr > 0.5:
            blur_transform = A.Compose([A.Blur(blur_limit=11, always_apply=False, p=1)])
            d_img = blur_transform(image=d_img)["image"]
            score = np.random.normal(0.25, 0.02)

        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample


class MyDataset_with_blur_384(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (384, 384), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)

        # apply blur or not
        p_aug = np.array([0.5, 0.5])
        prob_lr = np.random.choice([1, 0], p=p_aug.ravel())
        if prob_lr > 0.5:
            blur_transform = A.Compose([A.Blur(blur_limit=11, always_apply=False, p=1)])
            d_img = blur_transform(image=d_img)["image"]
            score = np.random.normal(0.25, 0.02)

        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample


class MyDataset_with_downscale(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)

        # apply blur or not
        p_aug = np.array([0.5, 0.5])
        prob_lr = np.random.choice([1, 0], p=p_aug.ravel())
        if prob_lr > 0.5:
            blur_transform = A.Compose(
                [
                    A.Downscale(
                        always_apply=False,
                        p=1.0,
                        scale_min=0.25,
                        scale_max=0.25,
                        interpolation=0,
                    )
                ]
            )
            d_img = blur_transform(image=d_img)["image"]
            score = np.random.normal(0.2, 0.02)

        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample


class MyDataset_caption(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])
        caption = self.df.comments[idx]
        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample, caption


class MyDataset_caption2(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d_img = cv2.imread(
            "/root/dacon/data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR
        )
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32")
        d_img = np.transpose(d_img, (2, 0, 1))
        img_path = self.df.img_path[idx]
        score = self.df.mos[idx]
        caption = self.df.comments[idx]

        return d_img, img_path, score, caption


class MyDataset_test(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        img_name = self.df.img_name[idx]

        sample = {"d_img_org": d_img, "img_name": img_name}
        if self.transform:
            sample = self.transform(sample)

        return sample


class MyDataset_384_test(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (384, 384), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        img_name = self.df.img_name[idx]

        sample = {"d_img_org": d_img, "img_name": img_name}
        if self.transform:
            sample = self.transform(sample)

        return sample


class MyDataset_256_test(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        img_name = self.df.img_name[idx]

        sample = {"d_img_org": d_img, "img_name": img_name}
        if self.transform:
            sample = self.transform(sample)

        return sample


class MyDataset_448_test(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (448, 448), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        img_name = self.df.img_name[idx]

        sample = {"d_img_org": d_img, "img_name": img_name}
        if self.transform:
            sample = self.transform(sample)

        return sample


class MyDataset_640_test(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (640, 640), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        img_name = self.df.img_name[idx]

        sample = {"d_img_org": d_img, "img_name": img_name}
        if self.transform:
            sample = self.transform(sample)

        return sample


class MyDataset_loss_check(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample, self.df.img_path[idx]


class MyDataset_384(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (384, 384), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample


class MyDataset_448(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (448, 448), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample


class MyDataset_640(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (640, 640), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample


class MyDataset_640_aug(torch.utils.data.Dataset):
    def __init__(self, csv_file, types):
        self.df = pd.read_csv(csv_file)
        if "mos" in self.df.columns:
            score = self.df.mos.to_numpy()
            score = self.normalization(score)
            self.score_list = list(score.astype("float").reshape(-1, 1))
        else:
            self.score_list = False
        self.types = types
        self.aug = self.make_aug(types)

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def make_aug(self, type):
        if type == "train":
            scales = [640]

            aug = A.Compose(
                [
                    A.HorizontalFlip(),
                    A.augmentations.geometric.resize.LongestMaxSize(max_size=scales),
                    A.PadIfNeeded(640, 640, border_mode=2),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ]
            )

            return aug

        else:
            scales = [640]

            aug = A.Compose(
                [
                    A.augmentations.geometric.resize.LongestMaxSize(max_size=scales),
                    A.PadIfNeeded(640, 640, border_mode=2),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ]
            )
            return aug

    def __getitem__(self, idx):
        d_img = Image.open("../../data" + self.df.img_path[idx][1:])
        d_img = d_img.convert("RGB")
        if self.score_list:
            score = self.score_list[idx].astype(dtype=np.float32)
        else:
            score = -1.0
        d_img = self.aug(image=np.array(d_img))["image"]

        if self.types != "test":
            sample = {"d_img_org": d_img, "score": score}
        else:
            img_name = self.df.img_name[idx]
            sample = {"d_img_org": d_img, "img_name": img_name}
        return sample


class MyDataset_640_crop_test(torch.utils.data.Dataset):
    def __init__(self, csv_file, types):
        self.df = pd.read_csv(csv_file)
        if "mos" in self.df.columns:
            score = self.df.mos.to_numpy()
            score = self.normalization(score)
            self.score_list = list(score.astype("float").reshape(-1, 1))
        else:
            self.score_list = False
        self.types = types
        self.small_aug = self.make_aug(small=True)
        self.crop_aug = self.make_aug(small=False)

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def make_aug(self, small):
        if small:
            aug = A.Compose(
                [
                    A.Resize(640, 640),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ]
            )
            return aug
        else:
            aug = A.Compose(
                [
                    A.RandomCrop(640, 640),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ]
            )
            return aug

    def __getitem__(self, idx):
        d_img = Image.open("../../data" + self.df.img_path[idx][1:])
        d_img = d_img.convert("RGB")
        if self.score_list:
            score = self.score_list[idx].astype(dtype=np.float32)
        else:
            score = -1.0
        h, w = d_img.size
        d_img_list = []
        if h < 640 or w < 640:
            for i in range(20):
                d_img_list.append(self.small_aug(image=np.array(d_img))["image"])
        else:
            for i in range(20):
                d_img_list.append(self.crop_aug(image=np.array(d_img))["image"])
        d_img = torch.stack(d_img_list, dim=0)
        if self.types != "test":
            sample = {"d_img_org": d_img, "score": score}
        else:
            img_name = self.df.img_name[idx]
            sample = {"d_img_org": d_img, "img_name": img_name}
        return sample


class MyDataset_384_crop(torch.utils.data.Dataset):
    def __init__(self, csv_file, types):
        self.df = pd.read_csv(csv_file)
        if "mos" in self.df.columns:
            score = self.df.mos.to_numpy()
            score = self.normalization(score)
            self.score_list = list(score.astype("float").reshape(-1, 1))
        else:
            self.score_list = False
        self.types = types
        self.aug = self.make_aug(types)

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def make_aug(self, type):
        if type == "train":
            scales = [384]

            aug = A.Compose(
                [
                    A.HorizontalFlip(),
                    A.PadIfNeeded(384, 384, border_mode=2),
                    A.RandomCrop(384, 384),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ]
            )

            return aug

        else:
            scales = [384]

            aug = A.Compose(
                [
                    A.PadIfNeeded(384, 384, border_mode=2),
                    A.RandomCrop(384, 384),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ]
            )
            return aug

    def __getitem__(self, idx):
        d_img = Image.open("../../data" + self.df.img_path[idx][1:])
        d_img = d_img.convert("RGB")
        if self.score_list:
            score = self.score_list[idx].astype(dtype=np.float32)
        else:
            score = -1.0

        d_img_list = []

        if self.types == "train" or self.types == "val":
            for i in range(5):
                d_img_list.append(self.aug(image=np.array(d_img))["image"])

        else:
            for i in range(20):
                d_img_list.append(self.aug(image=np.array(d_img))["image"])

        d_img = torch.stack(d_img_list, dim=0)
        if self.types != "test":
            sample = {"d_img_org": d_img, "score": score}
        else:
            img_name = self.df.img_name[idx]
            sample = {"d_img_org": d_img, "img_name": img_name}
        return sample


class MyDataset_384_aug(torch.utils.data.Dataset):
    def __init__(self, csv_file, types):
        self.df = pd.read_csv(csv_file)
        if "mos" in self.df.columns:
            score = self.df.mos.to_numpy()
            score = self.normalization(score)
            self.score_list = list(score.astype("float").reshape(-1, 1))
        else:
            self.score_list = False
        self.types = types
        self.aug = self.make_aug(types)

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def make_aug(self, type):
        if type == "train":
            scales = [384]

            aug = A.Compose(
                [
                    A.HorizontalFlip(),
                    A.augmentations.geometric.resize.LongestMaxSize(max_size=scales),
                    A.PadIfNeeded(384, 384, border_mode=2),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ]
            )

            return aug

        else:
            scales = [384]

            aug = A.Compose(
                [
                    A.augmentations.geometric.resize.LongestMaxSize(max_size=scales),
                    A.PadIfNeeded(384, 384, border_mode=2),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ]
            )
            return aug

    def __getitem__(self, idx):
        d_img = Image.open("../../data" + self.df.img_path[idx][1:])
        d_img = d_img.convert("RGB")
        if self.score_list:
            score = self.score_list[idx].astype(dtype=np.float32)
        else:
            score = -1.0
        d_img = self.aug(image=np.array(d_img))["image"]

        if self.types != "test":
            sample = {"d_img_org": d_img, "score": score}
        else:
            img_name = self.df.img_name[idx]
            sample = {"d_img_org": d_img, "img_name": img_name}
        return sample


class MyDataset_224_aug(torch.utils.data.Dataset):
    def __init__(self, csv_file, types):
        self.df = pd.read_csv(csv_file)
        if "mos" in self.df.columns:
            score = self.df.mos.to_numpy()
            score = self.normalization(score)
            self.score_list = list(score.astype("float").reshape(-1, 1))
        else:
            self.score_list = False
        self.types = types
        self.small_aug = self.make_aug(types, small=True)
        self.crop_aug = self.make_aug(types, small=False)

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def make_aug(self, type, small):
        if type == "train":
            if small:
                aug = A.Compose(
                    [
                        A.HorizontalFlip(),
                        A.Resize(224, 224),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                        ToTensorV2(),
                    ]
                )

                return aug
            else:
                aug = A.Compose(
                    [
                        A.HorizontalFlip(),
                        A.RandomCrop(224, 224),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                        ToTensorV2(),
                    ]
                )

                return aug

        else:
            if small:
                aug = A.Compose(
                    [
                        A.Resize(224, 224),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                        ToTensorV2(),
                    ]
                )

                return aug
            else:
                aug = A.Compose(
                    [
                        A.RandomCrop(224, 224),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                        ToTensorV2(),
                    ]
                )

                return aug

    def __getitem__(self, idx):
        d_img = Image.open("../../data" + self.df.img_path[idx][1:])
        d_img = d_img.convert("RGB")
        if self.score_list:
            score = self.score_list[idx].astype(dtype=np.float32)
        else:
            score = -1.0
        h, w = d_img.size

        d_img_list = []

        if h < 224 or w < 224:
            if self.types == "train":
                d_img = self.small_aug(image=np.array(d_img))["image"]
            else:
                for i in range(20):
                    d_img_list.append(self.small_aug(image=np.array(d_img))["image"])
        else:
            if self.types == "train":
                d_img = self.crop_aug(image=np.array(d_img))["image"]
            else:
                for i in range(20):
                    d_img_list.append(self.crop_aug(image=np.array(d_img))["image"])

        if self.types != "train":
            d_img = torch.stack(d_img_list, dim=0)

        if self.types != "test":
            sample = {"d_img_org": d_img, "score": score}
        else:
            img_name = self.df.img_name[idx]
            sample = {"d_img_org": d_img, "img_name": img_name}
        return sample


class MyDataset_384_sampling(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        e99 = self.df[self.df["weight"] == 1e99]
        self.df.loc[self.df["weight"] == 1e99, "weight"] = 0.0
        self.df = self.df.sample(n=20000 - 2898, weights="weight")
        self.df = pd.concat([self.df, e99])
        self.df.reset_index(inplace=True)
        self.df = self.df.drop("index", axis="columns")

        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (384, 384), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample


class MyDataset_256(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample


class MyDataset_crop(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        h, w, c = d_img.shape
        if h < 224 or w < 224:
            d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32")
        # d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        if self.transform:
            d_img = self.transform(image=d_img)["image"]
        sample = {"d_img_org": d_img, "score": score}

        return sample


class MyDataset_crop_val(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32")
        # d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        if self.transform:
            d_img = self.transform(image=d_img)["image"]
        sample = {"d_img_org": d_img, "score": score}

        return sample


class MyDataset_crop_test(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32")
        # d_img = np.transpose(d_img, (2, 0, 1))

        if self.transform:
            d_img = self.transform(image=d_img)["image"]

        img_name = self.df.img_name[idx]
        sample = {"d_img_org": d_img, "img_name": img_name}
        return sample


class MyDataset_crop_384(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        h, w, c = d_img.shape
        if h < 384 or w < 384:
            d_img = cv2.resize(d_img, (384, 384), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32")
        # d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        if self.transform:
            d_img = self.transform(image=d_img)["image"]
        sample = {"d_img_org": d_img, "score": score}

        return sample


class MyDataset_crop_val_384(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        h, w, c = d_img.shape
        if h < 384 or w < 384:
            d_img = cv2.resize(d_img, (384, 384), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32")
        # d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        transform_ = A.Compose(
            [
                A.RandomCrop(height=384, width=384),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(p=1),
            ]
        )

        image_ls = []
        for i in range(10):
            temp = transform_(image=d_img)["image"]
            image_ls.append(temp)

        image_ls = torch.stack(image_ls, dim=0)
        sample = {"d_img_ls": image_ls, "score": score}
        return sample


class MyDataset_crop_test_384(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        h, w, c = d_img.shape
        if h < 384 or w < 384:
            d_img = cv2.resize(d_img, (384, 384), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32")
        # d_img = np.transpose(d_img, (2, 0, 1))
        transform_ = A.Compose(
            [
                A.RandomCrop(height=384, width=384),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(p=1),
            ]
        )

        image_ls = []
        for _ in range(10):
            temp = transform_(image=d_img)["image"]
            image_ls.append(temp)

        image_ls = torch.stack(image_ls, dim=0)

        img_name = self.df.img_name[idx]
        sample = {"d_img_ls": image_ls, "img_name": img_name}
        return sample


# maxvit
class MyDataset_maxvit_train(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (512, 512), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32")
        score = self.score_list[idx]

        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample


class MyDataset_maxvit_val(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (512, 512), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32")
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])

        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample


class MyDataset_maxvit_test(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        score = self.df.mos.to_numpy()
        score = self.normalization(score)
        self.score_list = list(score.astype("float").reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def normalization(self, data):
        return data / 10

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (512, 512), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32")

        if self.transform:
            d_img = self.transform(image=d_img)["image"]

        img_name = self.df.img_name[idx]
        sample = {"d_img_org": d_img, "img_name": img_name}
        return sample


if __name__ == "__main__":
    a = MyDataset_384_sampling(
        "/root/dacon/data/train_only_mos/train_df_weight_fold1.csv", transform=None
    )
    b = MyDataset_384_sampling(
        "/root/dacon/data/train_only_mos/train_df_weight_fold1.csv", transform=None
    )
