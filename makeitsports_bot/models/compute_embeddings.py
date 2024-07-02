import torch
from makeitsports_bot.models.model import ViTImageSearchModel
import numpy as np
from sklearn.neighbors import NearestNeighbors
from makeitsports_bot.data.dataset import ImageRetrievalDataset
from makeitsports_bot.data.transforms import transform
from tqdm import tqdm
import os

def extract_embedding(image_data, fine_tuned_model):
    image = image_data.unsqueeze(0)
    with torch.no_grad():
        embedding = fine_tuned_model(image).cpu().numpy()
    return embedding


def load_fine_tuned_model():
    fine_tuned_model = ViTImageSearchModel()
    fine_tuned_model.load_state_dict(torch.load("results/model.pth"))
    fine_tuned_model.eval()
    return fine_tuned_model


def create_gallery(dataset, fine_tuned_model, save=True):
    gallery_embeddings = []
    for img_path, _ in tqdm(dataset):
        embedding = extract_embedding(img_path, fine_tuned_model)
        gallery_embeddings.append(embedding)
    gallery_embeddings = np.vstack(gallery_embeddings)
    if save:
        np.save("results/embeddings", gallery_embeddings)
    return gallery_embeddings


def search_image(query_image_path, gallery_embeddings, k=4):
    fine_tuned_model = load_fine_tuned_model()
    query_embedding = extract_embedding(query_image_path, fine_tuned_model)
    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean")
    neighbors.fit(gallery_embeddings)
    distances, indices = neighbors.kneighbors(query_embedding)
    return indices, distances


def create_gallery_embeddings(folder):  # noqa
    x = np.array([f"{folder}/{file}" for file in os.listdir(folder)])
    gallery_data = np.array([x, x])
    gallery_dataset = ImageRetrievalDataset(gallery_data, transform=transform)
    fine_tuned_model = load_fine_tuned_model()
    create_gallery(gallery_dataset, fine_tuned_model)
