import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from my_exp.exp.comparison import moving_average

matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
data = np.load('../data/dataset/PSM/PSM_test_data.npy')
dates = np.load('../data/dataset/PSM/PSM_test_date.npy')
labels = np.load('../data/dataset/PSM/PSM_test_label.npy')

# 截取前1000个数据点（可根据需要调整）
subset_size = 180
data_sub = data[:subset_size, :]
#dates_sub = dates[:subset_size]
labels_sub = labels[:subset_size]
dates_sub = np.arange(0, 180)

# 转换日期格式（根据实际情况调整格式）
# try:
#     dates_sub = pd.to_datetime(dates_sub)  # 自动转换常见日期格式
# except:
#     dates_sub = pd.to_datetime(dates_sub, format='%Y-%m-%d')  # 手动指定日期格式

# 数据归一化（按特征归一化到0-1范围）
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_sub)

# 创建画布
plt.figure(figsize=(15, 6))

# 绘制所有时序特征（使用半透明线条）
# for i in range(data_scaled.shape[1] - 10 ):
#     plt.axis('off')
#     plt.plot(dates_sub,
#              data_scaled[:, i],
#              color='blue',
#              alpha=0.08,   # 调整透明度控制线条叠加效果
#              linewidth=0.8)

# 检测异常区域边界
anomaly_regions = []
current_start = None

num_features = 6
feature_arr = [0, 2, 4, 6, 7, 10]
cols = 1  # 设置每行2个子图，可以根据实际需要调整
rows = (num_features + cols - 1) // cols  # 计算行数，确保所有特征都有图

for i, label in enumerate(labels_sub):
    if label == 1 and current_start is None:
        current_start = i
    elif label == 0 and current_start is not None:
        anomaly_regions.append((current_start, i-1))
        current_start = None
# 处理最后一个可能的异常段
if current_start is not None:
    anomaly_regions.append((current_start, len(labels_sub)-1))

# 创建画布和子图
fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
axes = axes.flatten()  # 将二维数组展平成一维，方便循环

# 绘制每个时序数据
for index, i in enumerate(feature_arr):
    ax = axes[index]  # 获取当前子图
    data = moving_average(data_scaled[:, i], 4)
    dates = dates_sub[:len(data)]
    ax.plot(dates, data, color='black', alpha=0.8, linewidth=2)
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # 设置标题和标签
    # ax.set_title(f'Feature {i + 1}', fontsize=12)
    # ax.set_xlabel('Timestamp', fontsize=10)
    # ax.set_ylabel('Normalized Value', fontsize=10)
    #ax.grid(alpha=0.3)
    # ax.axis('off')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # 绘制异常区域背景
    for start, end in anomaly_regions:
        ax.axvspan(dates_sub[start], dates_sub[end], color='red', alpha=0.3)

# 处理多余的子图
for i in range(num_features, len(axes)):
    fig.delaxes(axes[i])

# 调整布局
plt.tight_layout()
plt.savefig("1.png")


# for i, label in enumerate(labels_sub):
#     if label == 1 and current_start is None:
#         current_start = i
#     elif label == 0 and current_start is not None:
#         anomaly_regions.append((current_start, i-1))
#         current_start = None
# # 处理最后一个可能的异常段
# if current_start is not None:
#     anomaly_regions.append((current_start, len(labels_sub)-1))
#
# # 绘制异常区域背景
# for start, end in anomaly_regions:
#     plt.axvspan(dates_sub[start],
#                 dates_sub[end],
#                 color='red',
#                 alpha=0.3,
#                 label='Anomaly Region' if start == 0 else "")
#
# # 美化图表
# plt.title('Multivariate Time Series with Anomaly Detection', fontsize=14)
# plt.xlabel('Timestamp', fontsize=12)
# plt.ylabel('Normalized Value', fontsize=12)
# plt.grid(alpha=0.3)
#
# # 显示图例（仅显示异常区域）
# handles, labels = plt.gca().get_legend_handles_labels()
# unique_labels = dict(zip(labels, handles))
# plt.legend(unique_labels.values(), unique_labels.keys())
#
# plt.tight_layout()
# plt.savefig("smd.png")