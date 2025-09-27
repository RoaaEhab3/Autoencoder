
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
