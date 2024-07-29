import numpy as np
import itertools
import torch
import torch.nn as nn

class FoldNetDecoder(nn.Module):
    def __init__(self, num_features, num_points=2048, std=0.3):
        super(FoldNetDecoder, self).__init__()
        self.m = num_points  # Set number of points from config
        self.std = std
        grid_size = int(np.ceil(np.sqrt(self.m)))
        self.meshgrid = [[-std, std, grid_size], [-std, std, grid_size]]
        self.num_features = num_features
        self.folding1 = nn.Sequential(
            nn.Conv1d(self.num_features + 2, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(self.num_features + 3, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )

    def build_grid(self, batch_size):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = points[:self.m]  # Select first m points
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, x):
        x = x.unsqueeze(1).transpose(1, 2).repeat(1, 1, self.m)
        points = self.build_grid(x.shape[0]).transpose(1, 2)
        if x.is_cuda:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points), dim=1)

        folding_result1 = self.folding1(cat1)
        cat2 = torch.cat((x, folding_result1), dim=1)
        folding_result2 = self.folding2(cat2)
        output = folding_result2.transpose(1, 2)
        return output
