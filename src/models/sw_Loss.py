import torch
import torch.nn as nn
from torch.autograd import Variable


def minibatch_rand_projections(batchsize, dim, num_projections=1000):
    projections = torch.randn((batchsize, num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=2, keepdim=True))
    return projections

def compute_practical_moments_sw(x, y, num_projections=30, device="cuda", degree=2.0):
    """
    x, y: [batch_size, num_points, dim=3]
    num_projections: integer number
    """
    dim = x.size(2)
    batch_size = x.size(0)
    projections = minibatch_rand_projections(batch_size, dim, num_projections).to(device)

    xproj = x.bmm(projections.transpose(1, 2)).to(device)
    yproj = y.bmm(projections.transpose(1, 2)).to(device)

    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0]).to(device)
    _sort_pow_p_get_sum = torch.sum(torch.pow(torch.abs(_sort), degree), dim=2)

    first_moment = _sort_pow_p_get_sum.mean(dim=1)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment

class SWD(nn.Module):
    def __init__(self, num_projs, device="cuda"):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y):
        squared_sw_2, _ = compute_practical_moments_sw(x, y, num_projections=self.num_projs, device=self.device)
        loss = squared_sw_2.mean(dim=0)
        return loss