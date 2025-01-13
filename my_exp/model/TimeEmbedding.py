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