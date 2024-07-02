from makeitsports_bot.data.dataset import ImageRetrievalDataset
from torch.utils.data import DataLoader
from makeitsports_bot.losses.contrastiveloss import ContrastiveLoss
from makeitsports_bot.data.data import get_data_from_local, split_train_val_test
from makeitsports_bot.data.transforms import transform
from makeitsports_bot.models.model import ViTImageSearchModel
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def fine_tune_vit(epochs, batch_size):
    data = get_data_from_local()
    train_data, val_data, test_data = split_train_val_test(data, 0.2, 0.1)
    np.save("results/test_data", test_data)
    train_dataset = ImageRetrievalDataset(train_data, transform=transform)
    val_dataset = ImageRetrievalDataset(val_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ViTImageSearchModel()

    # logs
    log_dir = "./logs/"
    writer = SummaryWriter(log_dir=log_dir)

    # params
    criterion = ContrastiveLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    epochs = epochs

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch
            optimizer.zero_grad()

            input_embeddings = model(inputs)
            label_embeddings = model(labels)

            loss = criterion(input_embeddings, label_embeddings)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            writer.add_scalar(
                "Train Loss", loss.item(), epoch * len(train_loader) + batch_idx
            )

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Average Train Loss", avg_train_loss, epoch)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader)}")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_idx, batch in enumerate(val_loader):
                inputs, labels = batch
                input_embeddings = model(inputs)
                label_embeddings = model(labels)

                loss = criterion(input_embeddings, label_embeddings)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar("Validation Loss", avg_val_loss, epoch)
            print(f"Validation Loss: {val_loss/len(val_loader)}")

    torch.save(model.state_dict(), "results/model.pth")
