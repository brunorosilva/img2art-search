import os
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from img2art_search.data.dataset import ImageRetrievalDataset
from img2art_search.data.transforms import transform
from img2art_search.models.compute_embeddings import search_image


def predict(img: Image.Image) -> list:
    tmp_img_path = "tmp_img.png"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if img:
        img.save(tmp_img_path)
        pred_img = np.array([[tmp_img_path], [tmp_img_path]])
        pred_dataset = ImageRetrievalDataset(pred_img, transform=transform)
        pred_image_data = pred_dataset[0][0].unsqueeze(0).to(DEVICE)
        indices, distances = search_image(pred_image_data)
        results = []
        for index, distance in zip(indices, distances):
            buffered = BytesIO(index)
            image = Image.open(buffered)
            decoded_image_array = np.array(image)

            results.append(
                (
                    Image.fromarray(decoded_image_array),
                    str(distance),
                )
            )
        os.remove(tmp_img_path)
        return results
    else:
        return []
