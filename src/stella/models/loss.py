from torch import nn
from geomloss import SamplesLoss


class MMDLoss(nn.Module):
    def __init__(self, kernel="energy", blur=0.05, scaling=0.5, downsample=1):
        super().__init__()
        self.mmd_loss = SamplesLoss(loss=kernel, blur=blur, scaling=scaling)
        self.downsample = downsample

    def forward(self, input, target):
        input = input.reshape(-1, self.downsample, input.shape[-1])
        target = target.reshape(-1, self.downsample, target.shape[-1])

        return self.mmd_loss(input, target).mean()