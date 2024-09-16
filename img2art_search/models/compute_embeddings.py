import os

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

from img2art_search.constants import DEVICE
from img2art_search.data.dataset import ImageRetrievalDataset
from img2art_search.data.transforms import transform
from img2art_search.models.model import ViTImageSearchModel


def extract_embedding(image_data_batch, fine_tuned_model):
    image_data_batch = image_data_batch.to(DEVICE)
    with torch.no_grad():
        embeddings = fine_tuned_model(image_data_batch).cpu().numpy()
    return embeddings


def load_fine_tuned_model():
    fine_tuned_model = ViTImageSearchModel()
    fine_tuned_model.load_state_dict(torch.load("results/model.pth"))
    fine_tuned_model.eval()
    return fine_tuned_model


def create_gallery(img_dataset, fine_tuned_model, save=True):
    fine_tuned_model.to(DEVICE)
    gallery_embeddings = []
    gallery_dataloader = DataLoader(
        img_dataset, batch_size=100, num_workers=8, shuffle=False
    )
    for img_data, _ in tqdm(gallery_dataloader):
        embedding = extract_embedding(img_data, fine_tuned_model)
        gallery_embeddings.append(embedding)
    print(len(gallery_embeddings))
    gallery_embeddings = np.vstack(gallery_embeddings)
    if save:
        np.save("results/embeddings", gallery_embeddings)
    return gallery_embeddings


def search_image(query_image_path, gallery_embeddings, k=4):
    fine_tuned_model = load_fine_tuned_model()
    fine_tuned_model.to(DEVICE)
    query_embedding = extract_embedding(query_image_path, fine_tuned_model)
    neighbors = NearestNeighbors(n_neighbors=k, metric="cosine")
    neighbors.fit(gallery_embeddings)
    distances, indices = neighbors.kneighbors(query_embedding)
    return indices, distances


def create_gallery_embeddings(folder):  # noqa
    x = np.array([f"{folder}/{file}" for file in os.listdir(folder)])
    gallery_data = np.array([x, x])
    gallery_dataset = ImageRetrievalDataset(gallery_data, transform=transform)
    fine_tuned_model = load_fine_tuned_model()
    print(gallery_dataset)
    create_gallery(gallery_dataset, fine_tuned_model)
