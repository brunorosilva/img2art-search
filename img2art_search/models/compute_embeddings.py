import torch
import os
from pathlib import Path

from img2art_search.models.model import ViTImageSearchModel
import numpy as np
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from img2art_search.data.dataset import ImageRetrievalDataset
from img2art_search.data.transforms import transform
from tqdm import tqdm
import os
from sklearn.metrics.pairwise import cosine_distances


def extract_embedding(image_data_batch, fine_tuned_model, device):
    image_data_batch = image_data_batch.to(device)
    with torch.no_grad():
        embeddings = fine_tuned_model(image_data_batch).cpu().numpy()
    return embeddings

def create_gallery(dataset, fine_tuned_model, batch_size=256, save=True, save_path="results/embeddings.npy"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_tuned_model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    gallery_embeddings = []

    for idx, batch_images, _ in tqdm(dataloader):
        try:
            batch_embeddings = extract_embedding(batch_images, fine_tuned_model, device)
            gallery_embeddings.append(batch_embeddings)
        except Exception as e:
            print(e)
            print(idx)
    gallery_embeddings = np.vstack(gallery_embeddings)
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, gallery_embeddings)
    return gallery_embeddings
def load_fine_tuned_model():
    fine_tuned_model = ViTImageSearchModel()
    fine_tuned_model.load_state_dict(torch.load("results/model.pth"))
    fine_tuned_model.eval()
    return fine_tuned_model

def search_image(query_embedding, gallery_embeddings, k=5):
    distances = cosine_distances(query_embedding, gallery_embeddings)
    indices = np.argsort(distances, axis=1)[:, :k]
    sorted_distances = np.sort(distances, axis=1)[:, :k]
    return indices, sorted_distances


def create_gallery_embeddings(folder):  # noqa
    paths = sorted(Path(folder).iterdir(), key=os.path.getmtime)
    x = np.array(paths)
    gallery_data = np.array((x, x)).T
    gallery_dataset = ImageRetrievalDataset(gallery_data, transform=transform)
    print(gallery_data.shape)
    fine_tuned_model = load_fine_tuned_model()
    create_gallery(gallery_dataset, fine_tuned_model)
