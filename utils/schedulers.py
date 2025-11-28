from torch.optim.lr_scheduler import CosineAnnealingLR

def build_scheduler(optimizer, T_max=1500):
    return CosineAnnealingLR(optimizer, T_max=T_max)
