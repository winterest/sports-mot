"""Alignment Model
options: AffineRegression | PureSTN | ResSTN
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision


class AffineRegression(nn.Module):
    ####    INPUTS: x, y \in R^{n,3,512,512}
    ####    OUTPUTS: theta \in R^{n,2,3}
    def __init__(self, inc=6, p=124, only_alpha=False):
        super().__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(inc, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        if only_alpha:
            self.only_alpha = True
            self.fc_loc = nn.Sequential(
                nn.Linear(10 * p * p, 32), nn.ReLU(True), nn.Linear(32, 1)
            )

            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(
                torch.tensor([0], dtype=torch.float)
            )

        else:
            self.only_alpha = False
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(10 * p * p, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
            )

            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
            )

    # Spatial transformer network forward function
    def forward(self, x, y):
        device = x.device
        two_frames = torch.cat((x, y), 1)
        xs = self.localization(two_frames)
        n = xs.size(0)
        xs = xs.view(n, -1)
        if self.only_alpha:
            alpha = self.fc_loc(xs).view(-1)
            theta = torch.zeros((n, 2, 3)).to(device)
            theta[:, 0, 0], theta[:, 0, 1] = (
                torch.cos(alpha),
                -torch.sin(alpha),
            )
            theta[:, 1, 0], theta[:, 1, 1] = torch.sin(alpha), torch.cos(alpha)
        else:
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)
        return theta


class pureSTN(nn.Module):
    def __init__(self, inc=6, p=124):
        super().__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(inc, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * p * p, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x, y):
        two_frames = torch.cat((x, y), 1)
        xs = self.localization(two_frames)
        n = xs.size(0)
        xs = xs.view(n, -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # print(theta)
        # print(theta.size())

        return xs, theta

    def forward(self, x, y):
        # transform the input
        xs, theta = self.stn(x, y)
        return xs, theta


class ResSTN(nn.Module):
    def __init__(self, inc=6, p=124):
        super(ResSTN, self).__init__()

        # Spatial transformer localization-network
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        self.res_backbone = torchvision.models.resnet18(num_classes=6)
        self.res_backbone.conv1 = torch.nn.Conv2d(
            6,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

        # Initialize the weights/bias with identity transformation
        # self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.res_backbone.fc.weight.data.zero_()
        self.res_backbone.fc.bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x, y):
        two_frames = torch.cat((x, y), 1)
        theta = self.res_backbone(two_frames)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x, theta

    def forward(self, x, y):
        # transform the input
        x, theta = self.stn(x, y)
        return x, theta
