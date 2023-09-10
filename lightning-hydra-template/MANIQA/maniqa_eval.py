import os
import torch
import numpy as np
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from utils.process import ToTensor, Normalize
from models.maniqa import MANIQA
from tqdm import tqdm
from data.koniq10k.koniq10k import MyDataset
from scipy.stats import spearmanr, pearsonr
import pandas as pd

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = Config({
        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8
    })
model_path= "/root/dacon/lightning-hydra-template/logs/ckpt_koniq10k.pt"
net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint)
net = net.cuda()


test_transforms = transforms.Compose([Normalize(0.5, 0.5), ToTensor()])
test_dataset = MyDataset(csv_file='/root/dacon/data/train_df.csv', transform=test_transforms)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    num_workers=4,
    shuffle=False
)

with torch.no_grad():
    net.eval()
    name_list = []
    pred_list = []
    train_pred_epoch = []
    train_labels_epoch = [] 
    for batch in tqdm(test_loader):
        img, score = batch["d_img_org"], batch["score"].squeeze()
        img = img.cuda()
        score = score.cuda()
        pred = net(img)
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = score.data.cpu().numpy()
        train_pred_epoch = np.append(train_pred_epoch, pred_batch_numpy)
        train_labels_epoch = np.append(train_labels_epoch, labels_batch_numpy)
    rho_s, _ = spearmanr(
        np.squeeze(train_pred_epoch), np.squeeze(train_labels_epoch)
    )
    rho_p, _ = pearsonr(
        np.squeeze(train_pred_epoch), np.squeeze(train_labels_epoch)
    )
    print('SRCC :', rho_s)
    print('PLCC :', rho_p)
