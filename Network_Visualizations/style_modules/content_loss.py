import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def forward(self, content_weight, content_current, content_original):
        """

            Inputs:
            - content_weight: Scalar giving the weighting for the content loss.
            - content_current: features of the current image; this is a PyTorch Tensor of shape
              (1, C_l, H_l, W_l).
            - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

            Returns:
            - scalar content loss
            """

        diff = content_current - content_original
        return content_weight * torch.sum(diff ** 2)

