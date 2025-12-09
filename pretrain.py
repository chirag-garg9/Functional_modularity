import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as T
import os

from models import Autoencoder  # <- your model

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():

    # --------------------------
    # Dataset
    # --------------------------
    transform = T.Compose([
        T.Resize(64),
        T.ToTensor(),
    ])

    dataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)  # 0 workers for Windows

    # --------------------------
    # Model + Optimizer
    # --------------------------
    latent_dim = 10
    model = Autoencoder(latent_dim=latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Choose scheduler here
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    # Alternative: Step LR
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    criterion = nn.MSELoss()

    # --------------------------
    # Training settings
    # --------------------------
    max_epochs = 100
    patience = 10                  # stop if no improvement for X epochs
    best_loss = float("inf")
    patience_counter = 0

    print(f"Pretraining on FashionMNIST — epochs={max_epochs}, patience={patience}")

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0

        for x, _ in loader:
            x = x.to(device)

            x_recon, _ = model(x)
            loss = criterion(x_recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)

        # Step the scheduler
        scheduler.step()

        print(f"Epoch {epoch+1}/{max_epochs} — Loss: {epoch_loss:.5f} — LR: {scheduler.get_last_lr()[0]:.5f}")

        # --------------------------
        # Early stopping logic
        # --------------------------
        if epoch_loss < best_loss - 1e-5:   # require small improvement
            best_loss = epoch_loss
            patience_counter = 0

            # save best encoder weights
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.encoder.state_dict(), f"checkpoints/pretrained_encoder_fmnist_latent{latent_dim}.pth")

            print(f"✅ Improved | Saved best encoder (Loss: {best_loss:.5f})")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("⛔ Early stopping triggered.")
            break

    print(f"🏁 Training complete. Best loss: {best_loss:.5f}")


if __name__ == "__main__":
    main()
