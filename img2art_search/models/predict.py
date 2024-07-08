from img2art_search.data.dataset import ImageRetrievalDataset
from img2art_search.models.model import ViTImageSearchModel
import torch
from img2art_search.data.transforms import transform
from img2art_search.models.train import fine_tune_vit
from img2art_search.utils import inverse_transform_img
from img2art_search.models.compute_embeddings import (
    search_image,
    extract_embedding,
    load_fine_tuned_model,
)
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image


def predict(img: Image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fine_tuned_model()
    model.to(device)
    
    # Prepare the gallery dataset and load embeddings
    x = np.array([f"data/wikiart/{file}" for file in os.listdir("data/wikiart")])
    wikiart_data = np.array((x, x)).T
    wikiart_dataset = ImageRetrievalDataset(wikiart_data, transform=transform)
    gallery_embeddings = np.load("results/embeddings.npy")
    
    # Save the input image temporarily and create dataset
    tmp_img_path = "tmp_img.png"
    img.save(tmp_img_path)
    pred_img = np.array([[tmp_img_path], [tmp_img_path]]).T
    print(pred_img)
    pred_dataset = ImageRetrievalDataset(pred_img, transform=transform)
    
    # Extract embedding for the input image
    pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)
    pred_embedding = None
    for batch_images in pred_loader:
        pred_embedding = extract_embedding(batch_images[1][0].unsqueeze(0), model, device)
    
    # Search for similar images
    indices, distances = search_image(pred_embedding, gallery_embeddings)
    results = []
    print(indices, distances)
    for idx, _ in zip(indices[0], distances[0]):
        inv_tensor = inverse_transform_img(wikiart_dataset[idx][1]).cpu().numpy()
        results.append((inv_tensor, wikiart_data[idx][0].split("/")[-1].split(".jpg")[0]))
    print(results)
    # Clean up
    os.remove(tmp_img_path)
    
    return results