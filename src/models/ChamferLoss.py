import torch
import torch.nn as nn


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        bs, num_points_y, points_dim = y.size()  # Ensure batch size is consistent
        xx = x.pow(2).sum(dim=-1, keepdim=True)  # [bs, num_points_x, 1]
        yy = y.pow(2).sum(dim=-1, keepdim=True)  # [bs, num_points_y, 1]
        zz = torch.bmm(x, y.transpose(2, 1))  # [bs, num_points_x, num_points_y]
        rx = xx.expand(bs, num_points_x, num_points_y)
        ry = yy.transpose(2, 1).expand(bs, num_points_x, num_points_y)
        P = rx + ry - 2 * zz
        return P

    def forward(self, preds, gts):
        # Ensure that the input dimensions are correct
        assert preds.size(2) == 3 and gts.size(2) == 3, "Inputs must have shape [batch_size, num_points, 3]"
        assert preds.size(1) == gts.size(1), "Inputs must have the same number of points"

        P = self.batch_pairwise_dist(gts, preds)
        mins_pred_to_gt, _ = torch.min(P, 1)  # [bs, num_points_y]
        mins_gt_to_pred, _ = torch.min(P, 2)  # [bs, num_points_x]
        loss_pred_to_gt = torch.mean(mins_pred_to_gt)
        loss_gt_to_pred = torch.mean(mins_gt_to_pred)
        return loss_pred_to_gt + loss_gt_to_pred
