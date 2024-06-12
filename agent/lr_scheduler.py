from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
import numpy as np


class WarmupCosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, warmup_epochs, eta_min, eta_max, last_epoch=-1, verbose=True):
        self.warmup_epochs = warmup_epochs
        self.max_lr = eta_max
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, T_max, eta_min, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <= self.warmup_epochs:
            warmup_stage = self.last_epoch / self.warmup_epochs
            lrs = [base_lr + (self.max_lr - base_lr) * warmup_stage for base_lr in self.base_lrs]

        else:
            mid_stage = (self.last_epoch - self.warmup_epochs) / self.T_max
            lrs = [base_lr + (self.max_lr - base_lr) * (1 + np.cos(np.pi * mid_stage)) / 2 for base_lr in self.base_lrs]

        return lrs


class WarmupStepwiseLR(LRScheduler):
    def __init__(self, optimizer, boundaries=None, values=None, warmup_epochs=None, last_epoch=-1, verbose=False):
        self.boundaries = boundaries
        self.values = values
        self.warmup_epochs = warmup_epochs
        super(WarmupStepwiseLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Compute and return the next learning rate."""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_lr_start = self.values[0] * 0.01  # Starting from 1% of the first main LR
            warmup_lr_end = self.values[0]
            lr_scale = self.last_epoch / self.warmup_epochs
            warmup_lr = warmup_lr_start + (warmup_lr_end - warmup_lr_start) * lr_scale
            return [warmup_lr for _ in self.base_lrs]
        else:
            # Normal operation: pick the LR from the stepwise values
            idx = len([b for b in self.boundaries if b <= self.last_epoch])
            return [self.values[idx] for _ in self.base_lrs]
