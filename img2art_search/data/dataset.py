import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageRetrievalDataset(Dataset):
    def __init__(self, data: np.ndarray, transform: transforms.Compose) -> None:
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data[0])

    def __getitem__(self, idx: int) -> tuple:
        input_path, label_path = self.data.T[idx]
        input_image = Image.open(input_path).convert("RGB")
        label_image = Image.open(label_path).convert("RGB")

        # if self.transform:
        input_image = self.transform(input_image)
        label_image = self.transform(label_image)

        return input_image, label_image, input_path, label_path
