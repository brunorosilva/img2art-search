from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = 933120000


class ImageRetrievalDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(self.data[idx])
        input_path, label_path = self.data[idx]
        input_image = Image.open(input_path).convert("RGB")
        label_image = Image.open(label_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            label_image = self.transform(label_image)

        return idx, input_image, label_image
