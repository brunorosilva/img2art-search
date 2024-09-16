import os

import numpy as np
from PIL import Image

from img2art_search.constants import DEVICE
from img2art_search.data.dataset import ImageRetrievalDataset
from img2art_search.data.transforms import transform
from img2art_search.models.compute_embeddings import search_image
from img2art_search.utils import inverse_transform_img


def predict(img: Image):
    x = np.array([f"data/wikiart/{file}" for file in os.listdir("data/wikiart")])
    wikiart_data = np.array([x, x])
    wikiart_dataset = ImageRetrievalDataset(wikiart_data, transform=transform)
    gallery_embeddings = np.load("results/embeddings.npy")
    tmp_img_path = "tmp_img.png"
    if img:
        img.save(tmp_img_path)
        pred_img = np.array([[tmp_img_path], [tmp_img_path]])
        pred_dataset = ImageRetrievalDataset(pred_img, transform=transform)
        pred_image_data = pred_dataset[0][0].unsqueeze(0).to(DEVICE)
        indices, distances = search_image(pred_image_data, gallery_embeddings)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            inv_tensor = inverse_transform_img(wikiart_dataset[idx][1]).cpu().numpy()
            results.append(
                (
                    inv_tensor,
                    f'{wikiart_data[0][idx].split("/")[-1].split(".jpg")[0]} | {1-distance}', # noqa
                )
            )
        os.remove(tmp_img_path)
        return results
