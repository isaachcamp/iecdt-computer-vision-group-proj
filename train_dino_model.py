import logging
import os
import random
import ssl

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from pathlib import Path
from tqdm import tqdm

# Necessary to download pre-trained weights.
ssl._create_default_https_context = ssl._create_unverified_context

from define_dataset import CloudDataset, target_transform, split_data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.optim as optim
from transformers import Dinov2ForImageClassification


def validation(cfg, model, test_data_loader, criterion):
    model.eval()
    running_loss = 0
    num_batches = len(test_data_loader)
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_data_loader):
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            outputs = model(images).logits  # Pass images through the model
            loss = criterion(outputs, labels)  # Compute the loss
            running_loss += loss.item()

            if cfg.smoke_test and i == 2:
                num_batches = i + 1
                break

    return running_loss / num_batches


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    # Set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # wandb.init(mode="disabled")
    wandb.login(key=os.environ["WANDB_API_KEY"])
    # Generate ID to store and resume run.
    wandb_id = wandb.util.generate_id()
    wandb.init(
        id=wandb_id,
        resume="allow",
        project=cfg.wandb.project,
        group=cfg.name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.wandb.mode,
    )

    dataset = CloudDataset(
            Path(cfg.ground_truth_labels)  / 'hydrometeors_time_aligned_classes.csv', 
            Path(cfg.camera_a_images) / 'compressed_rectified_imgs', 
            transform=ToTensor(),
            target_transform=target_transform
    )

    train_loader, val_loader, test_loader = split_data(dataset, cfg.batch_size)

    logging.info("Loading DinoV2 model")
    model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base")
    if cfg.finetune:
    # Ensure we don't calculate gradients for the pre-trained weights as we're only finetuning the final layer
        logging.info("Freezing weights to only fine-tune final layer")
        for param in model.parameters():
            param.requires_grad = False
    model.classifier = nn.Linear(model.classifier.in_features, cfg.output_size)
    model.to(cfg.device)

    criterion = nn.BCEWithLogitsLoss()  # Use for classification with only two classes
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    logging.info("Starting training")
    # Training loop
    num_batches = len(train_loader)
    val_loss_min = np.inf

    for epoch in range(cfg.epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(cfg.device), labels.to(cfg.device)

            optimizer.zero_grad()  # Zero the gradients before the backward pass
            outputs = model(images).logits  # Pass images through the model
            # print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters

            total_loss += loss.item()  # Accumulate loss for this batch

        wandb.log({"loss/train": total_loss/num_batches})

        val_loss = validation(cfg, model, val_loader, criterion)
        wandb.log({"loss/val": val_loss})
        logging.info(
            f"Epoch {epoch}/{cfg.epochs} Average loss: {total_loss/num_batches:.4f}, Val Loss {val_loss:.4f}"
        )

        if cfg.smoke_test:
            break
    
    if val_loss < val_loss_min:
        logging.info(f"Validation loss decreased ({val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        val_loss_min = val_loss
        torch.save(model.state_dict(), Path(cfg.model_save_dir)/"model.pth")


if __name__ == "__main__":
    main()
