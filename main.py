from makeitsports_bot.models.predict import predict
from makeitsports_bot.models.train import fine_tune_vit
from makeitsports_bot.models.compute_embeddings import create_gallery_embeddings
import gradio as gr
import argparse

def make_interface():
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Gallery(label="Most similar images", height=256 * 3),
        live=True,
    )
    interface.launch()

def train(epochs, batch_size):
    fine_tune_vit(epochs, batch_size)

def create_gallery(gallery_path):
    create_gallery_embeddings(gallery_path)

def main():
    parser = argparse.ArgumentParser(description="Train or infer the ViT model for image-to-art search.")
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for training
    train_parser = subparsers.add_parser("train", help="Fine-tune the ViT model")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    # Subparser for inference
    _ = subparsers.add_parser("interface", help="Perform image-to-art search using the fine-tuned model")

    create_gallery_parser = subparsers.add_parser("gallery", help="Create new gallery from a path")
    create_gallery_parser.add_argument("--gallery_path", type=str, default="data/wikiart")
    args = parser.parse_args()

    if args.command == "train":
        train(args.epochs, args.batch_size)
    elif args.command == "interface":
        make_interface()
    elif args.command == "gallery":
        create_gallery(args.gallery_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
