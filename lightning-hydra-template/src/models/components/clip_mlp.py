import torch.nn as nn
import clip
from collections import OrderedDict

class clip_mlp(nn.Module):
    def __init__(self, vis_dim, device):
        super().__init__()
        # clip
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        for para in self.clip_model.parameters():
            para.requires_grad = False

        # mlp for image
        self.mlp_image = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("dropout1", nn.Dropout(0.5)),
            ("bn1", nn.BatchNorm1d(vis_dim//16)),
            ("linear2", nn.Linear(vis_dim // 16, 1)),
            ("sigmoid", nn.Sigmoid())
        ]))

        for m in self.mlp_image.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
        self.mlp_text = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("dropout1", nn.Dropout(0.5)),
            ("bn1", nn.BatchNorm1d(vis_dim//16)),
            ("linear2", nn.Linear(vis_dim // 16, 1)),
            ("relu", nn.ReLU(inplace=True)),
        ]))

        for m in self.mlp_text.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # text path
        caption = clip.tokenize("A photo of having score related to image quality.").to(self.device)
        caption = self.clip_model.encode_text(caption).float()
        caption = self.mlp_text(caption).squeeze()
        
        # image path
        x = self.clip_model.encode_image(x).float()
        x = self.mlp_image(x).squeeze()
        
        out = x * caption
        return out