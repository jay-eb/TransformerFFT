import torch

# 假设数据维度是 13000x25 (即有25个时序数据)
data = torch.randn(13000, 25)

# 设定 top_k 为保留的频率成分数量
top_k = 5

# 初始化存储季节性和趋势性的张量
x_season = torch.zeros_like(data)
x_trend = torch.zeros_like(data)

# 对每一列数据进行傅里叶变换，得到季节性与趋势性分量
for i in range(data.shape[1]):  # 对每一元数据（每一列）
    x = data[:, i]

    # 对该列数据应用傅里叶变换
    xf = torch.fft.rfft(x)

    # 计算频率的幅度
    freq = abs(xf)

    # 设定频率阈值，保留频率较大的部分
    freq[0] = 0  # 保证去除直流分量
    top_k_freq, top_list = torch.topk(freq, top_k)

    # 只保留前 top_k 个频率
    xf[freq <= top_k_freq.min()] = 0

    # 通过逆傅里叶变换得到季节性部分
    x_season[:, i] = torch.fft.irfft(xf)

    # 趋势部分即原数据减去季节性
    x_trend[:, i] = x - x_season[:, i]

# x_season 是季节性成分，x_trend 是趋势性成分
print(x_season.shape, x_trend.shape)
