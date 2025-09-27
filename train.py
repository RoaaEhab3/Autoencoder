import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import load_fashion_mnist
from model import VAE  # import VAE instead of AutoEncoder

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BETA = 1.0  # weight for KL divergence

# 1. Load data
train_data, test_data = load_fashion_mnist()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# 2. Initialize model, optimizer
model = VAE(latent_dim=16).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loss function (reconstruction + KL divergence)
def vae_loss(x, x_recon, mu, log_var):
    # Reconstruction loss (binary crossentropy or MSE)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="sum")

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + BETA * kl_loss, recon_loss, kl_loss


# 3. Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    recon_total = 0
    kl_total = 0

    for images, _ in train_loader:
        images = images.to(DEVICE)

        # Forward pass
        x_recon, mu, log_var = model(images)

        # Loss
        loss, recon_loss, kl_loss = vae_loss(images, x_recon, mu, log_var)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        recon_total += recon_loss.item()
        kl_total += kl_loss.item()

    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = recon_total / len(train_loader.dataset)
    avg_kl = kl_total / len(train_loader.dataset)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}"
    )

# 4. Save model
torch.save(model.state_dict(), "vae.pth")
print("✅ VAE model saved to vae.pth")
