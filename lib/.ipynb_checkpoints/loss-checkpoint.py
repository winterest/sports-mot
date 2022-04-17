# ------------------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import affine_grid, grid_sample


def union(x, y, union_fn="max"):
    if union_fn == "max":
        return torch.max(x, y)
    elif union_fn == "relu":
        relu = nn.ReLU()
        return relu(x - y) + y


class SegmentationLoss(nn.Module):
    def __init__(self, loss_type=nn.MSELoss, union_type="max", biloss=False):
        super(SegmentationLoss, self).__init__()
        self.loss = loss_type()
        self.union = union_type
        self.biloss = 1 if biloss else 0

    def forward(
        self,
        ref_frame,
        cur_frame,
        ref_mask,
        cur_mask,
        theta_ref_to_cur,
        theta_cur_to_ref,
    ):
        #

        grid_ref_to_cur_image = affine_grid(theta_ref_to_cur, ref_frame.size())
        ref_frame_to_cur = grid_sample(ref_frame, grid_ref_to_cur_image)
        grid_ref_to_cur_mask = affine_grid(theta_ref_to_cur, ref_mask.size())
        ref_mask_to_cur = grid_sample(ref_mask, grid_ref_to_cur_mask)

        grid_cur_to_ref_image = affine_grid(theta_cur_to_ref, ref_frame.size())
        cur_frame_to_ref = grid_sample(ref_frame, grid_cur_to_ref_image)
        grid_cur_to_ref_mask = affine_grid(theta_cur_to_ref, ref_mask.size())
        cur_mask_to_ref = grid_sample(ref_mask, grid_cur_to_ref_mask)

        fore_cur_and_ref = union(cur_mask, ref_mask_to_cur, self.union)
        fore_ref_and_cur = union(ref_mask, cur_mask_to_ref, self.union)

        back_cur_and_ref = 1 - fore_cur_and_ref
        back_ref_and_cur = 1 - fore_ref_and_cur

        loss_back = self.loss(
            ref_frame * back_ref_and_cur, cur_frame_to_ref * back_ref_and_cur
        ) + self.biloss * self.loss(
            cur_frame * back_cur_and_ref, ref_frame_to_cur * back_cur_and_ref
        )

        back_percentage = torch.sum(back_ref_and_cur) / torch.sum(
            torch.ones_like(back_ref_and_cur)
        )
        fore_percentage = torch.sum(fore_ref_and_cur) / torch.sum(
            torch.ones_like(fore_ref_and_cur)
        )

        loss_fore = self.loss(
            ref_frame * fore_ref_and_cur, cur_frame_to_ref * fore_ref_and_cur
        ) + self.biloss * self.loss(
            cur_frame * fore_cur_and_ref, ref_frame_to_cur * fore_cur_and_ref
        )

        loss_fore = loss_fore * back_percentage
        loss_back = loss_back * fore_percentage

        loss_total = loss_back - loss_fore

        return (
            loss_total,
            loss_back,
            loss_fore,
            fore_cur_and_ref,
            fore_ref_and_cur,
        )


class AlignmentLoss(nn.Module):
    """cycle_loss: "theta" | "frame"
    """

    def __init__(self, loss_type=F.mse_loss, cycle_loss="theta"):
        super().__init__()
        self.loss = loss_type
        self.cycle_loss = cycle_loss

    def forward(self, output0, theta0, output1, theta1, ref, target):
        """"""
        if self.cycle_loss == "theta":
            device = ref.device
            theta0 = torch.cat(
                (
                    theta0,
                    torch.tensor([0, 0, 1])
                    .repeat(theta0.size(0), 1, 1)
                    .to(device),
                ),
                dim=1,
            )
            theta1 = torch.cat(
                (
                    theta1,
                    torch.tensor([0, 0, 1])
                    .repeat(theta1.size(0), 1, 1)
                    .to(device),
                ),
                dim=1,
            )

            theta01 = torch.bmm(theta0, theta1)

            loss_cycle = (
                self.loss(
                    torch.eye(3).repeat(theta01.size(0), 1, 1).to(device),
                    theta01,
                )
                * 1000
            )
        else:
            loss_cycle = self.loss(output1, ref)
        loss_trans = self.loss(output0, target)
        loss = loss_trans + loss_cycle
        return loss, loss_trans, loss_cycle
