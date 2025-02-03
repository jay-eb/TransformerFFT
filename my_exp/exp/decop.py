import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 生成示例数据（替换为你的数据）
np.random.seed(42)
t = np.linspace(0, 10, 500)
trend = 0.5 * t  # 趋势项
periodic = 2 * np.sin(2 * np.pi * 2 * t)  # 周期项（2Hz）
noise = 0.8 * np.random.normal(size=500)
raw_data = trend + periodic + noise  # 原始数据 = 趋势 + 周期 + 噪声

# 执行FFT变换
fft_result = np.fft.fft(raw_data)
freq = np.fft.fftfreq(len(raw_data), d=(t[1]-t[0]))  # 计算频率

# 找到幅度第二大的频率成分（排除直流分量）
magnitudes = np.abs(fft_result)
sorted_indices = np.argsort(magnitudes)[::-1]  # 按幅度降序排列

# 排除0频率（直流分量），取第二大的非零频率
top_freq_index = sorted_indices[1]

# 创建周期项的频域表示（仅保留目标频率）
fft_periodic = np.zeros_like(fft_result)
fft_periodic[top_freq_index] = fft_result[top_freq_index]
fft_periodic[-top_freq_index] = fft_result[-top_freq_index]  # 保持对称性

# 逆FFT得到周期项时域信号
periodic_component = np.fft.ifft(fft_periodic).real

# 计算趋势项（原始数据 - 周期项）
trend_component = raw_data - periodic_component

# 绘制时序曲线图
plt.figure(figsize=(12, 6))
plt.plot(t, raw_data, label='原始数据', alpha=0.7)
plt.plot(t, periodic_component, label='周期项', linestyle='--')
plt.plot(t, trend_component, label='趋势项', linestyle='-.')
plt.xlabel('时间')
plt.ylabel('值')
plt.legend()
plt.savefig("fft1.png")

# 绘制FFT频域图
plt.figure(figsize=(12, 6))
positive_freq = freq[:len(freq)//2]
positive_magnitudes = magnitudes[:len(freq)//2]
plt.stem(positive_freq, positive_magnitudes, linefmt='b-', markerfmt=' ', basefmt=' ')
plt.scatter(freq[top_freq_index], magnitudes[top_freq_index],
            color='r', zorder=5, label='除掉直流外最大频率')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅值')
plt.xlim(0, 5)  # 显示前5Hz
plt.legend()
plt.savefig("fft2.png")