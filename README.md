---
title: Img2Art Search
emoji: 🐳
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
---
# img2art | Find Art That Matches Your Photos 🎨

**[🌐 Live Demo](https://brunorosilva.github.io/img2art-search/) | [🔍 Search Interface](https://brunorosilva.github.io/img2art-search/search/)**

> **Upload any image and discover visually similar masterpieces from 81,000+ artworks in the WikiArt collection.**

This project fine-tunes a Vision Transformer (ViT) model, pre-trained with `google/vit-base-patch32-224-in21k` weights, to perform image-to-art search across 81k artworks made available by [WikiArt](https://wikiart.org/).

## ✨ Features

- **Contextual Search** - Find art with similar themes, moods, and compositions
- **Expression Match** - Match portraits based on facial expressions and emotions
- **Shape & Form** - Discover art with similar shapes and visual structures
- **Modern Web Interface** - Beautiful, responsive frontend built with Next.js

## 🖼️ Examples

### Portrait Matching
| Photo | Matched Artwork |
|-------|-----------------|
| ![Portrait](https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=200&h=280&fit=crop) | **Lorenzo Lotto - Portrait Of A Young Man (1505)** - 98% match |
| ![Portrait](https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=200&h=280&fit=crop) | **Edgar Degas - Portrait Of Josephine Gaujelin (1867)** - 94% match |

### Landscape Matching
| Photo | Matched Artwork |
|-------|-----------------|
| ![Mountain](https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=200&h=140&fit=crop) | **Fyodor Vasilyev - Before A Thunderstorm (1869)** - 83% match |
| ![Waterfall](https://images.unsplash.com/photo-1433086966358-54859d0ed716?w=200&h=140&fit=crop) | **Gustave Courbet - A Family Of Deer In A Landscape With A Waterfall** - 78% match |

### Animals & Objects
| Photo | Matched Artwork |
|-------|-----------------|
| ![Cat](https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=200&h=140&fit=crop) | **Alexander Orlowski - Head Of A Cat (1823)** - 53% match |
| ![Dog](https://images.unsplash.com/photo-1517849845537-4d257902454a?w=200&h=140&fit=crop) | **Andy Warhol - Muhammad Ali 3** - 95% match |
| ![Flowers](https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=200&h=140&fit=crop) | **Claude Monet - Dahlias** - 85% match |

### Cities & Architecture
| Photo | Matched Artwork |
|-------|-----------------|
| ![Venice](https://images.unsplash.com/photo-1523531294919-4bcd7c65e216?w=200&h=140&fit=crop) | **Canaletto - Santi Giovanni E Paolo And The Scuola Di San Marco** - 95% match |
| ![Paris](https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=200&h=140&fit=crop) | **Ioannis Altamouras - No Name Seascape** - 60% match |

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [How it works](#how-it-works)
- [Interface](#interface)
- [Dataset](#dataset)
- [Training](#training)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project leverages the Vision Transformer (ViT) model architecture for image-to-art search. By fine-tuning the pre-trained ViT model on a custom images-to-artworks dataset, we create a model capable of matching any image to visually similar artworks from the [WikiArt](https://wikiart.org/) collection.

## Installation

1. Clone the repository:
```sh
git clone https://github.com/brunorosilva/img2art-search.git
cd img2art-search
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

1. Create a dataset matching images to artworks
2. Organize the images into appropriate directories for training and validation
3. Fine-tune the model
4. Create the gallery using WikiArt

### Training

Fine-tune the ViT model:
```sh
make train
```

### Inference via Gradio

Perform image-to-art search using the fine-tuned model:
```sh
make viz
```

### Recreate the WikiArt gallery
```sh
make wikiart
```

### Create new gallery

If you want to index new images to search, use:
```sh
poetry run python main.py gallery --gallery_path <your_path>
```

## Interface

### Web Application (Recommended)

The recommended way to use img2art is through the web interface:

- **[🌐 Homepage](https://brunorosilva.github.io/img2art-search/)** - Gallery showcase with example matches
- **[🔍 Search](https://brunorosilva.github.io/img2art-search/search/)** - Upload your own images to find matching artworks

The frontend is built with Next.js and features:
- Drag-and-drop image upload
- Real-time artwork matching
- Beautiful gallery showcase
- Responsive design for all devices

### Self-hosted Options

You can also run the interface locally:
- **Gradio Interface**: Run `make viz` for a local Gradio interface
- **Hugging Face Space**: [Demo on Hugging Face](https://huggingface.co/spaces/chicelli/img2art-search)

## Dataset

The fine-tuning dataset derives from 1k examples of images and artworks. Images are split into training, validation, and test sets.

WikiArt is indexed using the same process, except that there's no expected result. Each artwork is mapped to itself and the model is used as a feature extractor. The gallery embeddings are saved to Pinecone for fast similarity search.

## Training

The training script fine-tunes the ViT model on the prepared dataset. Key steps include:

1. Loading the pre-trained `google/vit-base-patch32-224-in21k` weights
2. Preparing the dataset and data loaders
3. Fine-tuning the model using a custom training loop
4. Saving the model to the models folder

## Contributing

There are some topics I'd appreciate help with:

1. **Expanding the gallery** - The current gallery has 81k artworks. The complete WikiArt catalog has 250k+ artworks, so I'd like to reach at least 300k
2. **Text-based search with CLIP** - Add optional text search terms
3. **Video2art search** - Match video frames to artworks
4. **Performance improvements** - Optimize search speed and accuracy
5. **New ideas** - Open issues with suggestions for improvements

## License

The source code for the site is licensed under the MIT license, which you can find in the MIT-LICENSE.txt file.

All graphical assets are licensed under the Creative Commons Attribution 3.0 Unported License.
