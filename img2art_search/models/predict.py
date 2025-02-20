import os
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from img2art_search.data.dataset import ImageRetrievalDataset
from img2art_search.data.transforms import transform
from img2art_search.models.compute_embeddings import search_image


def predict(img: Image.Image) -> list:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if img:
        img = img.convert("RGB")
        pred_image_data = transform(img).unsqueeze(0).to(DEVICE)
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
        return results
    else:
        return []
