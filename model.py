import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Reparameterization trick
# -------------------------
class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # compute size after conv layers for flatten
        self.flatten = nn.Flatten()
        self.latent_dim = latent_dim

        # lazy initialization
        self.fc_mu = None
        self.fc_logvar = None
        self.sampling = Sampling()

    def forward(self, x):
        x = F.relu(self.conv1(x))   # [B, 32, H/2, W/2]
        x = F.relu(self.conv2(x))   # [B, 64, H/4, W/4]
        x = F.relu(self.conv3(x))   # [B, 128, H/8, W/8]

        # Save shape for decoder
        self.shape_before_flattening = x.shape[1:]  # (C, H, W)
        x = self.flatten(x)

        if self.fc_mu is None:  # lazy init
            in_features = x.shape[1]
            self.fc_mu = nn.Linear(in_features, self.latent_dim)
            self.fc_logvar = nn.Linear(in_features, self.latent_dim)
            self.fc_mu.to(x.device)
            self.fc_logvar.to(x.device)

        z_mean = self.fc_mu(x)
        z_log_var = self.fc_logvar(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z

# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):
    def __init__(self, shape_before_flattening, latent_dim=32):
        super().__init__()
        C, H, W = shape_before_flattening
        self.fc = nn.Linear(latent_dim, C * H * W)
        self.C, self.H, self.W = C, H, W

        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.C, self.H, self.W)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.final(x))
        return x

# -------------------------
# VAE Model
# -------------------------
class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32, image_size=28):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        # dummy forward to initialize decoder correctly
        dummy = torch.zeros(1, in_channels, image_size, image_size)
        _, _, _ = self.encoder(dummy)
        self.decoder = Decoder(self.encoder.shape_before_flattening, latent_dim)

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def loss_function(self, x, reconstruction, z_mean, z_log_var, beta=1.0):
        # reconstruction loss
        recon_loss = F.binary_cross_entropy(reconstruction, x, reduction="sum") / x.size(0)
        # KL divergence
        kl_loss = -0.5 * torch.mean(torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1))
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss
