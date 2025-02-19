import base64
import os
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from img2art_search.data.dataset import ImageRetrievalDataset
from img2art_search.data.transforms import transform
from img2art_search.models.model import ViTImageSearchModel
from img2art_search.utils import (
    get_or_create_pinecone_index,
    get_pinecone_client,
    inverse_transform_img,
)


def extract_embedding(image_data_batch, fine_tuned_model):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    image_data_batch = image_data_batch.to(DEVICE)
    with torch.no_grad():
        embeddings = fine_tuned_model(image_data_batch).cpu().numpy()
    return embeddings


def load_fine_tuned_model():
    fine_tuned_model = ViTImageSearchModel()
    fine_tuned_model.load_state_dict(torch.load("models/model.pth"))
    fine_tuned_model.eval()
    return fine_tuned_model


def create_gallery(
    img_dataset: ImageRetrievalDataset,
    fine_tuned_model: ViTImageSearchModel,
    save: bool = True,
) -> list:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    fine_tuned_model.to(DEVICE)
    gallery_embeddings = []
    gallery_dataloader = DataLoader(
        img_dataset, batch_size=batch_size, num_workers=1, shuffle=False
    )

    pc = get_pinecone_client()
    gallery_index = get_or_create_pinecone_index(pc)
    try:
        count = 0
        for img_data, _, img_name, _ in tqdm(gallery_dataloader):
            data_objects = []
            batch_embedding = extract_embedding(img_data, fine_tuned_model)
            gallery_embeddings.append(batch_embedding)
            for idx, embedding in enumerate(batch_embedding):
                image = Image.fromarray(
                    inverse_transform_img(img_data[idx]).numpy().astype("uint8"), "RGB"
                )
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data_objects.append(
                    {
                        "id": str(count),
                        "values": embedding.tolist(),
                        "metadata": {
                            "image": img_base64,
                            "name": img_name[idx]
                            .split("/")[-1]
                            .replace(".jpg", "")
                            .replace(".jpeg", "")
                            .replace(".png", "")
                            .replace(".JPG", "")
                            .replace(".JPEG", "")
                            .replace("-", " ")
                            .replace("_", " - ")
                            .title(),
                        },
                    }
                )
                count += 1
            gallery_index.upsert(vectors=data_objects)
    except Exception as e:
        print(f"Error creating gallery: {e}")

    if save:
        np.save("models/embeddings", gallery_embeddings)
    return gallery_embeddings


def search_image(query_image_path: str, k: int = 4) -> tuple:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    fine_tuned_model = load_fine_tuned_model()
    fine_tuned_model.to(DEVICE)
    query_embedding = extract_embedding(query_image_path, fine_tuned_model)
    pc = get_pinecone_client()
    index = get_or_create_pinecone_index(pc)
    response = index.query(
        vector=[query_embedding.tolist()[0]], top_k=k, include_metadata=True
    )
    distances = []
    results = []
    for obj in response["matches"]:
        result = base64.b64decode(obj.metadata["image"])
        results.append(result)
        distances.append(
            str(round(obj["score"], 2) * 100) + " " + str(obj.metadata["name"])
        )

    return results, distances


def create_gallery_embeddings(folder: str) -> None:
    x = np.array([f"{folder}/{file}" for file in os.listdir(folder)])
    gallery_data = np.array([x, x])
    gallery_dataset = ImageRetrievalDataset(gallery_data, transform=transform)
    fine_tuned_model = load_fine_tuned_model()
    create_gallery(gallery_dataset, fine_tuned_model)
