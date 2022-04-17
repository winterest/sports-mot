"""Alignment and Segmentation Model

"""
import torch
from torch import nn
from segmentation import UNet
from aligning import AffineRegression


class Model(nn.Module):
    """define the super model for combination of segmentation and alignment"""

    def __init__(
        self,
        unet_inc=3,
        unet_feat=32,
        unet_outc=1,
        align_inc=6,
        align_p=124,
        only_alpha=False,
    ):
        # super(Model, self).__init__()
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.unet = UNet(
            in_channels=unet_inc,
            out_channels=unet_outc,
            init_features=unet_feat,
        )
        self.align = AffineRegression(
            inc=align_inc, p=align_p, only_alpha=only_alpha
        )

    def forward(self, ref_frame, cur_frame):
        """
        input: ref_frame: f_{t-s} cur_frame: f_t
        output: theta_{t-s to t}, theta_{t to t-s}, mask_{t-s}, mask_t
        """
        theta_ref_to_cur = self.align(ref_frame, cur_frame)
        theta_cur_to_ref = self.align(cur_frame, ref_frame)
        ref_mask = self.unet(ref_frame)
        cur_mask = self.unet(cur_frame)

        return theta_ref_to_cur, theta_cur_to_ref, ref_mask, cur_mask
