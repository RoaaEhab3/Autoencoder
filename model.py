import torch
import torch.nn as nn

# -------------------------------
# Encoder
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=1, feature_dims=[64, 128, 256, 512]):
        """
        Args:
            in_channels: input channels (1 for grayscale FashionMNIST)
            feature_dims: list of feature sizes for each stage
        """
        super(Encoder, self).__init__()

        # Stage 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, feature_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Conv2d(feature_dims[0], feature_dims[0], kernel_size=3, stride=2, padding=1)

        # Stage 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_dims[0], feature_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[1]),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Conv2d(feature_dims[1], feature_dims[1], kernel_size=3, stride=2, padding=1)

        # Stage 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_dims[1], feature_dims[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[2]),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Conv2d(feature_dims[2], feature_dims[2], kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_dims[2], feature_dims[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[3]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.down1(x1)

        x2 = self.conv2(x)
        x = self.down2(x2)

        x3 = self.conv3(x)
        x = self.down3(x3)

        x4 = self.bottleneck(x)

        # skips from shallow → deep
        skips = [x3, x2, x1]
        return x4, skips


class Decoder(nn.Module):
    def __init__(self, feature_dims=[512, 256, 128, 64], out_channels=1):
        super(Decoder, self).__init__()

        # Stage 1
        self.up1 = nn.ConvTranspose2d(feature_dims[0], feature_dims[1], kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_dims[1]*2, feature_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[1]),
            nn.ReLU(inplace=True)
        )

        # Stage 2
        self.up2 = nn.ConvTranspose2d(feature_dims[1], feature_dims[2], kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_dims[2]*2, feature_dims[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[2]),
            nn.ReLU(inplace=True)
        )

        # Stage 3
        self.up3 = nn.ConvTranspose2d(feature_dims[2], feature_dims[3], kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_dims[3]*2, feature_dims[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[3]),
            nn.ReLU(inplace=True)
        )

        # Final output
        self.final_conv = nn.Conv2d(feature_dims[3], out_channels, kernel_size=1)

    def forward(self, x, skips):
        # Stage 1
        x = self.up1(x)
        if x.shape[2:] != skips[0].shape[2:]:  # align sizes
            x = nn.functional.interpolate(x, size=skips[0].shape[2:], mode="nearest")
        x = torch.cat([x, skips[0]], dim=1)
        x = self.conv1(x)

        # Stage 2
        x = self.up2(x)
        if x.shape[2:] != skips[1].shape[2:]:
            x = nn.functional.interpolate(x, size=skips[1].shape[2:], mode="nearest")
        x = torch.cat([x, skips[1]], dim=1)
        x = self.conv2(x)

        # Stage 3
        x = self.up3(x)
        if x.shape[2:] != skips[2].shape[2:]:
            x = nn.functional.interpolate(x, size=skips[2].shape[2:], mode="nearest")
        x = torch.cat([x, skips[2]], dim=1)
        x = self.conv3(x)

        return self.final_conv(x)

class AutoEncoder(nn.Module):
    def __init__(self, in_channels=1, feature_dims=[64, 128, 256, 512]):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, feature_dims=feature_dims)
        self.decoder = Decoder(feature_dims=list(reversed(feature_dims)), out_channels=in_channels)

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        reconstructed = self.decoder(bottleneck, skips)
        return reconstructed
