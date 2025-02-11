import matplotlib.pyplot as plt
import numpy as np


def plot_anomaly_comparison(true_data, my_scores, transformer_scores, labels):
    timesteps = np.arange(len(true_data))

    fig, ax1 = plt.subplots(figsize=(15, 6))

    # 绘制真实数据曲线
    ax1.plot(timesteps, true_data, label='True Data', color='blue', alpha=0.8)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 绘制异常区域
    start = None
    for i, label in enumerate(labels):
        if label == 1 and start is None:
            start = i
        elif label == 0 and start is not None:
            ax1.axvspan(start, i - 1, color='red', alpha=0.3)
            start = None
    # 处理最后一个可能的异常区间
    if start is not None:
        ax1.axvspan(start, len(labels) - 1, color='red', alpha=0.3)

    # 创建第二个y轴用于异常分数
    ax2 = ax1.twinx()
    ax2.plot(timesteps, my_scores, label='My Model', color='green', alpha=0.8)
    ax2.plot(timesteps, transformer_scores, label='Transformer', color='orange', alpha=0.8)
    ax2.set_ylabel('Anomaly Score', color='black')
    ax2.tick_params(axis='y')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Anomaly Detection Comparison')
    plt.tight_layout()
    plt.show()


# 在你的测试代码最后调用该函数
def test(self):
    # ... [你的原有测试代码] ...

    # 获取必要数据
    true_data = test_src[:, 0]  # 假设取第一个特征作为展示
    my_scores = test_score.reshape(-1)
    labels = test_label.reshape(-1)

    # 需要获取Transformer模型的分数（这里需要你自行实现）
    transformer_scores = get_transformer_scores()  # 替换为实际获取方法

    # 调用绘图函数
    plot_anomaly_comparison(true_data, my_scores, transformer_scores, labels)