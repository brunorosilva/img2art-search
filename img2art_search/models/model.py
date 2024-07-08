from transformers import ViTModel
from torch import nn
import torch

class ViTImageSearchModel(nn.Module):
    def __init__(self, pretrained_model_name="google/vit-base-patch32-224-in21k"):
        super(ViTImageSearchModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

    def forward(self, x):  # noqa
        outputs = self.vit(pixel_values=x)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        return cls_hidden_state
