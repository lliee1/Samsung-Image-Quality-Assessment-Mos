import numpy as np
import timm
import torch
from timm.models.vision_transformer import Block
import torch.nn as nn
from timm.layers import NormMlpClassifierHead, get_norm_layer
from functools import partial


class Maxvit(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('maxvit_base_tf_512.in1k', pretrained=True)
        final_norm_layer = partial(get_norm_layer('layernorm2d'), eps=1e-6)
        self.model.head = NormMlpClassifierHead(
                        768,
                        1,
                        hidden_size=768,
                        pool_type="avg",
                        drop_rate=0.,
                        norm_layer=final_norm_layer,
                    )

    def forward(self, x):
        x = self.model(x)
        return x            
   

