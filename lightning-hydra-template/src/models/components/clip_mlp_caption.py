import torch.nn as nn
import clip
from collections import OrderedDict
import torch


class clip_mlp_caption(nn.Module):
    def __init__(self, vis_dim, device):
        super().__init__()
        # clip
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        for para in self.clip_model.parameters():
            para.requires_grad = False

        # mlp for image
        self.mlp_image = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim * 2, vis_dim * 2 // 16)),
                    ("gelu", nn.GELU()),
                    ("linear2", nn.Linear(vis_dim * 2 // 16, 1)),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

        for m in self.mlp_image.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.mlp_text = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim)),
                    ("gelu", nn.GELU()),
                    ("linear2", nn.Linear(vis_dim, vis_dim)),
                ]
            )
        )

        for m in self.mlp_text.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, caption):
        # text path
        caption = clip.tokenize(caption).to(self.device)
        caption = self.clip_model.encode_text(caption).float()
        caption_after = self.mlp_text(caption)
        caption = caption + 1e-4 * caption_after

        # image path
        x = torch.cat([x, caption], dim=1)
        x = self.clip_model.encode_image(x).float()
        x = self.mlp_image(x).squeeze()

        out = caption * x
        return out
