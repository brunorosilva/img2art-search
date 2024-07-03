from img2art_search.data.dataset import ImageRetrievalDataset
from img2art_search.data.transforms import transform
from img2art_search.models.train import fine_tune_vit
from img2art_search.utils import inverse_transform_img
from img2art_search.models.compute_embeddings import (
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
        results.append((inv_tensor, wikiart_data[0][idx].split("/")[-1].split(".jpg")[0]))
    os.remove(tmp_img_path)
    return results
