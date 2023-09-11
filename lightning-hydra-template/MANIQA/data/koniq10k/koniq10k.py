import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



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
            d_img = self.transform(image=d_img)['image']
        sample = {"d_img_org": d_img, "score": score}
        
        return sample

class MyDataset_crop_val(torch.utils.data.Dataset):
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
        if h < 224 or w < 224:
            d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32")
        # d_img = np.transpose(d_img, (2, 0, 1))
        score = self.score_list[idx]
        # score = np.array(self.df.mos[idx])
        
        transform_ = A.Compose([
            A.RandomCrop(height=224,width=224),
            A.Normalize(mean = 0.5, std= 0.5),
            ToTensorV2(p=1)       
        ])
        
        image_ls = []
        for i in range(10):
            temp = transform_(image=d_img)['image']
            image_ls.append(temp)

        image_ls = torch.stack(image_ls, dim=0)
        sample = {"d_img_ls": image_ls, "score": score}
        return sample
    
class MyDataset_crop_test(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d_img = cv2.imread("../../data" + self.df.img_path[idx][1:], cv2.IMREAD_COLOR)
        h, w, c = d_img.shape
        if h < 224 or w < 224:
            d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32")
        # d_img = np.transpose(d_img, (2, 0, 1))
        transform_ = A.Compose([
            A.RandomCrop(height=224,width=224),
            A.Normalize(mean = 0.5, std= 0.5),
            ToTensorV2(p=1)       
        ])
        
        image_ls = []
        for _ in range(10):
            temp = transform_(image=d_img)['image']
            image_ls.append(temp)
                
        image_ls = torch.stack(image_ls, dim=0)
        
        img_name = self.df.img_name[idx]
        sample = {"d_img_ls": image_ls, "img_name": img_name}
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
            d_img = self.transform(image=d_img)['image']
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
        
        transform_ = A.Compose([
            A.RandomCrop(height=384,width=384),
            A.Normalize(mean = 0.5, std= 0.5),
            ToTensorV2(p=1)       
        ])
        
        image_ls = []
        for i in range(10):
            temp = transform_(image=d_img)['image']
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
        transform_ = A.Compose([
            A.RandomCrop(height=384,width=384),
            A.Normalize(mean = 0.5, std= 0.5),
            ToTensorV2(p=1)       
        ])
        
        image_ls = []
        for _ in range(10):
            temp = transform_(image=d_img)['image']
            image_ls.append(temp)
                
        image_ls = torch.stack(image_ls, dim=0)
        
        img_name = self.df.img_name[idx]
        sample = {"d_img_ls": image_ls, "img_name": img_name}
        return sample