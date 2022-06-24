import torch.nn as nn
import torch


example = torch.rand(1, 10, 15)
print(example)

pow_ = torch.pow(example, 2)
print(pow_)
print(0.6843 ** 2)