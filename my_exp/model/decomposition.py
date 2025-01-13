import torch
from torch import nn

from my_exp.model.block import SpatialTemporalTransformerBlock, TemporalTransformerBlock, DecompositionBlock
from my_exp.model.embedding import DataEmbedding, PositionEmbedding, TimeEmbedding


class DataEncoder(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, block_num, head_num, dropout):
        super(DataEncoder, self).__init__()
        self.data_embedding = DataEmbedding(model_dim, feature_num)
        self.position_embedding = PositionEmbedding(model_dim)

        self.encoder_blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            self.encoder_blocks.append(
                SpatialTemporalTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dp)
            )

    def forward(self, x):
        x = self.data_embedding(x) + self.position_embedding(x)

        for block in self.encoder_blocks:
            x = block(x)

        return x


class TimeEncoder(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, time_num, block_num, head_num, dropout):
        super(TimeEncoder, self).__init__()
        self.time_embed = TimeEmbedding(model_dim, time_num)

        self.encoder_blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            self.encoder_blocks.append(
                TemporalTransformerBlock(model_dim, ff_dim, atten_dim, head_num, dp)
            )

    def forward(self, x):
        x = self.time_embed(x)

        for block in self.encoder_blocks:
            x = block(x)

        return x


class DynamicDecomposition(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, time_num, block_num, head_num, dropout,
                 d):
        super(DynamicDecomposition, self).__init__()
        self.data_encoder = DataEncoder(window_size, model_dim, ff_dim, atten_dim, feature_num, block_num,
                                        head_num, dropout)
        self.time_encoder = TimeEncoder(model_dim, ff_dim, atten_dim, time_num, block_num, head_num, dropout)
        self.period_encoder = TimeEncoder(model_dim, ff_dim, atten_dim, time_num, block_num, head_num, dropout)

        self.decomposition_blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            self.decomposition_blocks.append(
                DecompositionBlock(model_dim, ff_dim, atten_dim, feature_num, head_num, dp)
            )

    def forward(self, data, period, time):
        residual = data.clone()

        data = self.data_encoder(data)
        time = self.time_encoder(time)
        period = self.period_encoder(period)
        stable = torch.zeros_like(residual).to(data.device)

        for block in self.decomposition_blocks:
            tmp_stable = block(data, period, time)
            stable = stable + tmp_stable

        return stable
