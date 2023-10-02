import torch
import numpy as np
import cv2
import pandas as pd


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

        sample = {"d_img_org": d_img, "score": score}

        if self.transform:
            sample = self.transform(sample)
        return sample
