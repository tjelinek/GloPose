from dataclasses import dataclass

import torch


@dataclass
class CameraIntrinsics:
    pass


class FlowLoss:

    def __int__(self):
        pass

    def masked_flow_loss(self, flow_pred: torch.Tensor, flow_true: torch.Tensor, mask: torch.Tensor):
        with torch.no_grad:
            masked_flow_pred = torch.mul(flow_pred, mask)
            masked_flow_true = torch.mul(flow_true, mask)

            masked_mse_loss = torch.nn.MSELoss(masked_flow_pred, masked_flow_true)

            loss = masked_mse_loss

        return loss
