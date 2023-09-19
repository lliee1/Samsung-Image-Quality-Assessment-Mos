import torch.nn as nn
import clip
from collections import OrderedDict
import torch.nn.functional as F


class liqe(nn.Module):
    def __init__(self, device):
        super().__init__()
        # clip
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False, )

    def forward(self, x, text):
        batch_size = x.size(0)
        num_patch = x.size(1)

        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        logits_per_image, logits_per_text = self.clip_model.forward(x, text)

        logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
        logits_per_text = logits_per_text.view(-1, batch_size, num_patch)

        logits_per_image = logits_per_image.mean(1)
        logits_per_text = logits_per_text.mean(2)

        logits_per_image = F.softmax(logits_per_image, dim=1)

        return logits_per_image, logits_per_text
