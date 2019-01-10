import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_dim, output_shape):
        """
        Args:
            input_dim: the dimension of the input space
            output_shape: the shape of the image to be generated
                grayscale only!

        """
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_shape = output_shape

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 128*(self.output_shape[0]//2)*(self.output_shape[1]//2)),
            nn.BatchNorm1d(128*(self.output_shape[0]//2)*(self.output_shape[1]//2)), nn.ReLU()
        )

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, X):
        X = self.fc(X)
        X = X.view(-1, 128, self.output_shape[0]//2, self.output_shape[1]//2)
        X = self.upconv(X)
        return X


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(128*(self.input_shape[0]//4)*(self.input_shape[1]//4), 1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2),
            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.conv(X)
        X = X.view(-1, 128*(self.input_shape[0]//4)*(self.input_shape[1]//4))
        X = self.fc(X)
        return X


if __name__ == '__main__':
    pass