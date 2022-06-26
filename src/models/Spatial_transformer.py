import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """Lernable Spatial transformer module.
    Allows spatial manipulation of data within the network
    """
    def __init__(self):
        super(SpatialTransformer, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ReLU(inplace=True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(in_features=32 * 14 * 2, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def forward(self, x):
        x_tr = self.localization(x)
        x_tr = x_tr.view(-1, 32*14*2)
        theta = self.fc_loc(x_tr)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SpatialTransformer().to(device)
    input_ = torch.Tensor(2, 3, 24, 94).to(device)
    output = model(input_)
    print('Output shape is: ', output.shape)
