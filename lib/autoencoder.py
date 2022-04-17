import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, inc=3):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(inc, 8, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        return x


class Decoder(nn.Module):
    def __init__(self, outc=3):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3))
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 3))
        self.conv3 = nn.ConvTranspose2d(16, 8, kernel_size=(3, 3))
        self.conv4 = nn.ConvTranspose2d(8, outc, kernel_size=(3, 3))
        self.sigmoid_activation = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.sigmoid_activation(x) * 2 - 1  # * 255
        return x


class Autoencoder(nn.Module):
    def __init__(self, inc=3, outc=3):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(inc)
        self.decoder = Decoder(outc)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
