import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    def forward(self, img, tv_weight):
        """

            Inputs:
            - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
            - tv_weight: Scalar giving the weight w_t to use for the TV loss.

            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img weighted by tv_weight.
            """

        h_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
        v_diff = img[:, :, 1:, :] - img[:, :, :-1, :]

        loss = torch.sum(h_diff ** 2) + torch.sum(v_diff ** 2)
        loss = tv_weight * loss

        return loss
