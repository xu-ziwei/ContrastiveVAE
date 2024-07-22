import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024):
        super(DGCNN, self).__init__()
        self.k = k
        self.emb_dims = emb_dims

        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = self.get_graph_feature(x, k=self.k)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.get_graph_feature(x, k=self.k)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.get_graph_feature(x, k=self.k)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.get_graph_feature(x, k=self.k)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_max_pool1d(x.max(dim=-1, keepdim=False)[0], 1).view(batch_size, -1)
        x = F.relu(self.bn5(self.conv5(x.unsqueeze(2)))).view(batch_size, -1)

        return x

    def get_graph_feature(self, x, k=20):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        x = x.permute(0, 2, 1)

        idx = knn(x, k=k)
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points

        idx = idx + idx_base
        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        x = x.permute(0, 2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx