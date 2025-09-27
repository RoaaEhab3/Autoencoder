# /content/Autoencoder/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 16):
        super().__init__()
        # input: (B, in_channels, 28, 28)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)  # -> 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)           # -> 7x7
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var


# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        # inverse of encoder: start from latent -> project to (64,7,7) -> upsample to 28x28
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # -> 14x14
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)   # -> 28x28

    def forward(self, z: torch.Tensor):
        x = self.fc(z)
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))  # output in [0,1]
        return x


# -------------------------
# VAE
# -------------------------
class VAE(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 16):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    @staticmethod
    def loss_function(x: torch.Tensor, x_recon: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, beta: float = 1.0):
        """
        Returns: (total_loss, recon_loss, kl_loss) where recon_loss and kl_loss are averaged per-batch.
        - recon_loss uses binary cross entropy (sum over pixels, divided by batch size)
        - kl_loss is mean KL per batch
        """
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss


# Optional helper to load weights (path can be absolute or relative)
def load_model(path: str, device: str = None) -> VAE:
    """
    Load a saved VAE state dict from `path`. Returns the model on the requested device.
    Example:
        model = load_model("vae.pth", device="cuda")
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # instantiate with default latent_dim=16; if you used a different size, change below
    model = VAE(latent_dim=16).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

