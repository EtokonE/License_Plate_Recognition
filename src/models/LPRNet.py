import torch
import torch.nn as nn
from typing import Sequence


class SmallBasicBlock(nn.Module):
    """Implementation of Small Basic Block

    Args:
        in_channels (int): Number of channels in the input feature map
        out_channels (int): Number of channels in the output of Basic Block
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(SmallBasicBlock, self).__init__()
        intermediate_channels = out_channels // 4

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    """Licence Plate Recognition Network

    Args:
        class_num (int): Corresponds to the number of all possible characters
        dropout_prob (float): Probability of an element to be zeroed in nn.Dropout
        out_indices (Sequence[int]): Indices of layers, where we want to extract feature maps and use it 
            for embedding in global context
    """
    def __init__(self, 
                class_num: int, 
                dropout_prob: float, 
                out_indices: Sequence[int]):
        super(LPRNet, self).__init__()

        self.class_num = class_num
        self.out_indices = out_indices

        self.backbone = nn.Sequential(
            
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(), # -> extract feature map (2)
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),

            SmallBasicBlock(in_channels=64, out_channels=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(), # -> extract feature map (6)
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            
            SmallBasicBlock(in_channels=64, out_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            SmallBasicBlock(in_channels=256, out_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(), # -> extract feature map (13)
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            nn.Dropout(dropout_prob),
            
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),
            nn.BatchNorm2d(num_features=self.class_num),
            nn.ReLU(), # -> extract feature map (22)
        )
        # in_channels - sum of all channels in extracted feature maps (see the marks above)
        self.container = nn.Conv2d(in_channels=64+128+256+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        extracted_feature_maps = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in self.out_indices:
                extracted_feature_maps.append(x)

        global_contex_emb = list()
        for i, feature_map in enumerate(extracted_feature_maps):
            if i in (0, 1):
                feature_map = nn.AvgPool2d(kernel_size=5, stride=5)(feature_map)
            if i == 2:
                feature_map = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(feature_map)
            f_pow = torch.pow(feature_map, 2)
            f_mean = torch.mean(f_pow)
            feature_map = torch.div(feature_map, f_mean)
            global_contex_emb.append(feature_map)

        x = torch.cat(global_contex_emb, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        return logits

if __name__ == '__main__':
    from torchsummary import summary
    from src.config.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    
    lprnet = LPRNet(class_num=len(cfg.CHARS.LIST), 
                    dropout_prob=cfg.LPRNet.DROPOUT,
                    out_indices=cfg.LPRNet.OUT_INDEXES)
    print(lprnet)

    summary(lprnet, (3, 24, 94), device='cpu')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_ = torch.Tensor(2, 3, 24, 94).to(device)
    output = lprnet(input_)
    print('Output shape is: ', output.shape)
    #print(output[0])
    #print(type(output))
    from src.tools.utils import BeamDecoder, GreedyDecoder
    beam_decoder = BeamDecoder()
    preds = output.cpu().detach().numpy()
    print(beam_decoder.decode(preds, cfg.CHARS.LIST))
    greedy_decoder = GreedyDecoder()
    print(greedy_decoder.decode(preds, cfg.CHARS.LIST))

