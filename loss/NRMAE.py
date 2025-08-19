import torch

def nrmae(pred, target, norm_by='range', eps=1e-8):
    """
    pred, target: Tensor, shape [B, C, H, W] or [H, W]
    norm_by: 'range' or 'mean'
    """
    mae = torch.mean(torch.abs(pred - target))
    
    if norm_by == 'range':
        norm = torch.max(target) - torch.min(target)
    elif norm_by == 'mean':
        norm = torch.mean(target)
    else:
        raise ValueError("norm_by must be 'range' or 'mean'")
    
    return mae / (norm + eps)
