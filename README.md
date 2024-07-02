# (WIP) MakeItSports Bot Image-to-Art Search

This project fine-tunes a Vision Transformer (ViT) model, pre-trained with "google/vit-base-patch32-224-in21k" weights and fine tuned with the style of [ArtButMakeItSports](https://www.instagram.com/artbutmakeitsports/), to perform image-to-art search across 81k artworks made available by [WikiArt](https://wikiart.org/).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Inference](#inference)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project leverages the Vision Transformer (ViT) model architecture for the task of image-to-art search. By fine-tuning the pre-trained ViT model on a custom dataset derived from the Instagram account [ArtButMakeItSports](https://www.instagram.com/artbutmakeitsports/), we aim to create a model capable of matching images (but not only) to corresponding artworks, being able to search for any of the images on [WikiArt](https://wikiart.org/).

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/brunorosilva/makeitsports-bot.git
    cd makeitsports-bot
    ```

2. Install poetry:
    ```sh
    pip install poetry
    ```

3. Install using poetry:
    ```sh
    poetry install
    ```

## How it works

### Dataset Preparation

1. Download images from the [ArtButMakeItSports](https://www.instagram.com/artbutmakeitsports/) Instagram account.
2. Organize the images into appropriate directories for training and validation.

### Training

1. Fine-tune the ViT model:
    ```sh
    poetry run python main.py train --epochs 50 --batch_size 32
    ```

### Inference via Gradio

1. Perform image-to-art search using the fine-tuned model:
    ```sh
    poetry run python main.py interface
    ```

### Create new gallery

1. If you want to index new images to search, use:
    ```sh
    poetry run python main.py gallery --gallery_path <your_path>
    ```

## Dataset

The dataset derives from 1k images from the Instagram account [ArtButMakeItSports](https://www.instagram.com/artbutmakeitsports/). Images are downloaded and split into training, validation and test sets. Each image is paired with its corresponding artwork for training purposes, if you want this dataset just ask me stating your usage.

WikiArt is indexed using the same process, except that there's no expected result. So each artwork is mapped to itself and the embeddings are saved as a numpy file (will be changed to chromadb in the future).

## Training

The training script fine-tunes the ViT model on the prepared dataset. Key steps include:

1. Loading the pre-trained "google/vit-base-patch32-224-in21k" weights.
2. Preparing the dataset and data loaders.
3. Fine-tuning the model using a custom training loop.