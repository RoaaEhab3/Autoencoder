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

