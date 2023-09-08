import torch.nn as nn
import clip
from collections import OrderedDict

class clip_mlp(nn.Module):
    def __init__(self, vis_dim, device):
        super().__init__()
        # clip
        self.clip_model, self.preprocess = clip.load("ViT-B/16", device=device)
        for para in self.clip_model.parameters():
            para.requires_grad = False

        # mlp
        self.mlp = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("bn1", nn.BatchNorm1d(vis_dim//16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, 1))
        ]))

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.clip_model.encode_image(x).float()
        x = self.mlp(x).squeeze()
        
        return x