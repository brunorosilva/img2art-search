import os
from typing import Any

import torch
from pinecone import Pinecone, ServerlessSpec

from img2art_search.data.transforms import inversetransform

pinecone_api_key = os.environ["PINECONE_API_KEY"]

def inverse_transform_img(img: torch.Tensor) -> torch.Tensor:
    inv_tensor = inversetransform(img)
    tensor_image = (inv_tensor * 255).byte()
    return tensor_image.permute(1, 2, 0)


def get_pinecone_client() -> Pinecone:
    pc = Pinecone(api_key=pinecone_api_key)
    return pc


def get_or_create_pinecone_index(
    pc: Pinecone, index_name: str = "img2art-search", embeddings_dim: int = 768
) -> Any:
    indexes_names = [index.name for index in pc.list_indexes()]
    if index_name not in indexes_names:
        pc.create_index(
            name=index_name,
            dimension=embeddings_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)

    return index
