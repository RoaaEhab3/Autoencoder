import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import load_fashion_mnist
from model import AutoEncoder

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load data
train_data, test_data = load_fashion_mnist()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# 2. Initialize model, loss, optimizer
model = AutoEncoder(in_channels=1).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, _ in train_loader:
        images = images.to(DEVICE)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# 4. Save the model
torch.save(model.state_dict(), "autoencoder.pth")
print("✅ Model saved to autoencoder.pth")
