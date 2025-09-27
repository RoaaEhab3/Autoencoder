#important libraries
import torch
import torch.nn as nn
import torch.optim as optim

# important variables
IMAGE_SIZE = 32
CHANNELS = 1
BATCH_SIZE = 100
BUFFER_SIZE = 1000
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 2
EPOCHS = 3
LEARNING_RATE = 0.001

# Encoder class
class Encoder(nn.Module):
    def __init__(self, in_channels=3, feature_dims=[64, 128, 256, 512]):
        """
        Args:
            in_channels: number of input channels (3 for RGB)
            feature_dims: must match reversed decoder feature_dims
                          [64, 128, 256, 512] → last is bottleneck
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
        """
        Returns:
            bottleneck: deepest feature map
            skips: [skip1, skip2, skip3] for decoder
        """
        # Stage 1
        x1 = self.conv1(x)
        x = self.down1(x1)

        # Stage 2
        x2 = self.conv2(x)
        x = self.down2(x2)

        # Stage 3
        x3 = self.conv3(x)
        x = self.down3(x3)

        # Bottleneck
        x4 = self.bottleneck(x)

        # Decoder expects [skip1, skip2, skip3] from shallow → deep
        skips = [x3, x2, x1]
        return x4, skips

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):











class Decoder(nn.Module):
    def __init__(self, feature_dims=[512, 256, 128, 64], out_channels=3):
        super(Decoder, self).__init__()
        
        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(feature_dims[0], feature_dims[1], kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_dims[1]*2, feature_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[1]),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(feature_dims[1], feature_dims[2], kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_dims[2]*2, feature_dims[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[2]),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(feature_dims[2], feature_dims[3], kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_dims[3]*2, feature_dims[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[3]),
            nn.ReLU(inplace=True)
        )
        
        # Final reconstruction layer
        self.final_conv = nn.Conv2d(feature_dims[3], out_channels, kernel_size=1)

    def forward(self, x, skips):
        """
        Args:
            x: bottleneck feature (batch, feature_dims[0], H/8, W/8)
            skips: list of skip connections from encoder
                   [skip1, skip2, skip3] with matching feature sizes
        """
        # stage 1
        x = self.up1(x)
        x = torch.cat([x, skips[0]], dim=1)  # concat with encoder skip
        x = self.conv1(x)
        
        # stage 2
        x = self.up2(x)
        x = torch.cat([x, skips[1]], dim=1)
        x = self.conv2(x)
        
        # stage 3
        x = self.up3(x)
        x = torch.cat([x, skips[2]], dim=1)
        x = self.conv3(x)
        
        # final RGB reconstruction
        x = self.final_conv(x)
        return x
