import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # (3,256,256) -> (8,128,128)
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  # (8,128,128) -> (16,64,64)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # (16,64,64) -> (32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (32,32,32) -> (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (64,16,16) -> (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (128,8,8) -> (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (64,16,16) -> (32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # (32,32,32) -> (16,64,64)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # (16,64,64) -> (8,128,128)
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),  # (8,128,128) -> (3,256,256)
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)   # (3,256,256) -> (8,128,128)
        x2 = self.conv2(x1)  # (8,128,128) -> (16,64,64)
        x3 = self.conv3(x2)  # (16,64,64) -> (32,32,32)
        x4 = self.conv4(x3)  # (32,32,32) -> (64,16,16)
        x5 = self.conv5(x4)  # (64,16,16) -> (128,8,8)

        # Decoder forward pass
        d1 = self.deconv1(x5)  # (128,8,8) -> (64,16,16)
        d2 = self.deconv2(d1)  # (64,16,16) -> (32,32,32)
        d3 = self.deconv3(d2)  # (32,32,32) -> (16,64,64)
        d4 = self.deconv4(d3)  # (16,64,64) -> (8,128,128)
        output = self.deconv5(d4)  # (8,128,128) -> (3,256,256)

        return output
    