import torch
import torch.nn as nn
import torch.nn.functional as F
from zmq import device


class SpatialTransformer(nn.Module):
    """Lernable Spatial transformer module.
    Allows spatial manipulation of data within the network
    """
    def __init__(self):
        super(SpatialTransformer, self).__init__()

        self.locallization = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ReLU(inplace=True)
        )

        self.fc_affine = nn.Sequential(
            nn.Linear(in_features=32 * 14 * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )

        self.fc_affine[2].weight.data.zero_()
        self.fc_affine[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))


    def forward(self, x):
        x_tr = self.locallization(x)
        x_tr = x_tr.view(-1, 32*14*2)
        theta = self.fc_affine(x_tr)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SpatialTransformer().to(device)
    imput = torch.Tensor(2, 3, 24, 94).to(device)
    output = model(input)
    print('Output shape is: ', output.shape, '\n', 'Output: ', output)