from easydict import EasyDict as edict

from . import loss

def get_losses_by_config(cfg, device0):
    assert len(cfg.LOSSES) > 0, 'At least one loss function must be used for training!' 
    losses = edict() # dictionary of all losses used in training or validation
    losses_weights = edict() # weights of losses (in case they are computed in a weighted combination)

    for loss_type in cfg.LOSSES: # "loss" must be a name of a loss
        loss_dict = cfg.LOSSES[loss_type]
        assert 'NAME' in loss_dict, 'each loss must provide a function name' # really need it? TODO
        params = loss_dict.PARAMS if 'PARAMS' in loss_dict else {}

        # all loss functions must lie in the "losses/loss.py" 
        # NAME - must be a name of the loss class
        losses[loss_type] = eval("loss."+f'{loss_dict.NAME}')(**params) 
        losses_weights[loss_type] = 1. if 'WEIGHT' not in loss_dict else loss_dict.WEIGHT # default is 1.

    for loss_type in losses:
        losses[loss_type] = losses[loss_type].to(device0)
    return losses, losses_weights