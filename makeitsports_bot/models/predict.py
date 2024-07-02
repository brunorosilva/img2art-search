from makeitsports_bot.data.dataset import ImageRetrievalDataset
from makeitsports_bot.data.transforms import transform
from makeitsports_bot.models.train import fine_tune_vit
from makeitsports_bot.utils import inverse_transform_img
from makeitsports_bot.models.compute_embeddings import (
    search_image,
    create_gallery,
    load_fine_tuned_model,
)
import numpy as np
import os
from PIL import Image


def predict(img: Image):
    x = np.array([f"data/wikiart/{file}" for file in os.listdir("data/wikiart")])
    wikiart_data = np.array([x, x])
    wikiart_dataset = ImageRetrievalDataset(wikiart_data, transform=transform)
    gallery_embeddings = np.load("results/embeddings.npy")
    tmp_img_path = "tmp_img.png"
    img.save(tmp_img_path)
    pred_img = np.array([[tmp_img_path], [tmp_img_path]])
    pred_dataset = ImageRetrievalDataset(pred_img, transform=transform)
    indices, distances = search_image(pred_dataset[0][0], gallery_embeddings)
    results = []
    for idx, _ in zip(indices[0], distances[0]):
        inv_tensor = inverse_transform_img(wikiart_dataset[idx][1]).cpu().numpy()
        results.append(inv_tensor)
    return results
