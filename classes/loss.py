from torch import nn
import torch



class LocalLoss(nn.Module):

    def __init__(self):
        super(LocalLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        

    def forward(self, pred, y):
        i = pred.argmax(dim=1) // 10
        
        loss = self.ce_loss(pred.view(-1, 81, 9)[i], y[i])
        
        return loss
    

class GlobalLoss(nn.Module):

    def __init__(self):
        super(GlobalLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        

    def forward(self, pred, y):

        loss = self.ce_loss(pred, y.view(-1))
        
        return loss


class DoubleLoss(nn.Module):

    def __init__(self):
        super(DoubleLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self,
                cell_pred: torch.Tensor,    # [batch_size, 81]
                update_pred: torch.Tensor,  # [batch_size, 9]
                cell_y: torch.Tensor,       # [batch_size, 1]
                update_y: torch.Tensor      # [batch_size, 1]
               ) -> torch.Tensor:
        """
        Computes the double cross-entropy loss.

        Args:
        - cell_pred (torch.Tensor): Predictions for cell.
        - update_pred (torch.Tensor): Predictions for update.
        - cell_y (torch.Tensor): True labels for cell.
        - update_y (torch.Tensor): True labels for update.

        Returns:
        - torch.Tensor: The computed loss.
        """

        loss = self.ce_loss(cell_pred, cell_y.view(-1)) + self.ce_loss(update_pred, update_y.view(-1))

        return loss