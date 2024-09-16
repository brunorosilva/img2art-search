import torch

BASE_PATH = "data/artmakeitsports"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
