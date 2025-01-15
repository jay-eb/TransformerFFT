import numpy as np
import matplotlib.pyplot as plt

# 1. 加载数据
data_file = "train_data.npy"  # 数据文件路径
date_file = "train_date.npy"  # 时间文件路径
data = np.load(data_file)  # (num_timesteps, num_features)
dates = np.load(date_file)  # (num_timesteps,)

# 检查数据形状
print(f"Data shape: {data.shape}")  # 特征值 (num_timesteps, num_features)
print(f"Dates shape: {dates.shape}")  # 时间信息 (num_timesteps,)

# 检查时间数据
print("Sample dates:", dates[:5])  # 打印前5个时间

# 2. 确保时间步数和数据行数匹配
assert data.shape[0] == dates.shape[0], "时间和特征数据长度不一致！"

# 3. 绘制多元时序图
plt.figure(figsize=(14, 7))  # 调整画布大小

num_features = data.shape[1]
for i in range(num_features):
    plt.plot(dates, data[:, i], label=f"Feature {i+1}")  # 每个特征绘制一条曲线

# 4. 格式化图表
plt.title("Multivariate Time Series with Dates")
plt.xlabel("Time")
plt.ylabel("Feature Values")
plt.legend(loc="best")  # 显示图例
plt.grid(True)  # 添加网格
plt.tight_layout()  # 自动调整布局
plt.xticks(rotation=45)  # x轴标签旋转45度，防止重叠
plt.show()  # 显示图形
