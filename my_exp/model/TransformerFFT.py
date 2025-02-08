import torch
from torch import nn

from my_exp.model.decomposition import DynamicDecomposition


class TransformerFFT(nn.Module):
    def __init__(self, time_steps, beta_start, beta_end, window_size, model_dim, ff_dim, atten_dim, feature_num,
                 time_num, block_num, head_num, dropout, device, d, t, act):
        super(TransformerFFT, self).__init__()

        self.device = device
        self.window_size = window_size
        self.t = t

        self.dynamic_decomposition = DynamicDecomposition(
            window_size=window_size,
            model_dim=model_dim,
            ff_dim=ff_dim,
            atten_dim=atten_dim,
            feature_num=feature_num,
            time_num=time_num,
            block_num=block_num,
            head_num=head_num,
            dropout=dropout,
            d=d,
            act=act
        )

    def forward(self, trend, period, time, p=0):
        trend_s, period_s = self.dynamic_decomposition(trend, period, time)
        return trend_s, period_s
