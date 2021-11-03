import torch.nn as nn
import torch as th


class JointsLoss(nn.Module):

    def __init__(self, criterion='L2'):
        super(JointsLoss, self).__init__()
        self.criterion = criterion

    def forward(self, output, target, target_mask=None):

        batch_size, num_joints = output.size(0), output.size(1)
        output = output.reshape(-1,2)
        target = target.reshape(-1,2)

        if target_mask is not None:
            target_mask = target_mask.view(-1)
            output = output[target_mask]
            target = target[target_mask]
            num_samples = target_mask.sum()
        else:
            num_samples = batch_size * num_joints

        if self.criterion == 'L2':
            loss = ((output - target)**2).sum() / num_samples
        else:
            assert False # TODO
            
        return loss


class MPJPE(nn.Module):
    def __init__(self, pose_in_m=True):
        super(MPJPE, self).__init__()
        self.pose_in_m = pose_in_m

    def forward(self, output, target, target_mask=None):
        batch_size = output.size(0)

        output = output.reshape(batch_size,-1,3)
        target = target.reshape(batch_size,-1,3)

        num_joints = output.size(1)
        
        if self.pose_in_m:
            output = output * 1000.
            target = target * 1000.
        

        loss = th.sqrt(((output - target)**2).sum(dim=2)).sum() / num_joints / batch_size
        return loss
