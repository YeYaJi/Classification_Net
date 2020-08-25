import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


class All_affine_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.layers1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layers2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layers3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, input):
        la_1 = F.relu(self.layers1(input))
        la_2 = F.relu(self.layers2(la_1))
        out = self.layers3(la_2)
        return out
if __name__ == "__main__":
    model = All_affine_Net(28 * 28, 300, 100, 10)
    print(model)