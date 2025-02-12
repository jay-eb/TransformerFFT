import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_anomaly_comparison(true_data, my_scores, labels, pic_name):
    timesteps = np.arange(len(true_data))

    fig, ax1 = plt.subplots(figsize=(15, 6))

    # 绘制真实数据曲线
    ax1.plot(timesteps, true_data, label='Raw Data', color='black', alpha=0.8)
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('原始值', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

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
    smoothed_scores = moving_average(my_scores, 3)
    timesteps_smooth = timesteps[:len(smoothed_scores)]
    ax2.plot(timesteps_smooth, smoothed_scores, label='Transformer', color='green', alpha=0.8)
    # ax2.plot(timesteps, transformer_scores, label='Transformer', color='orange', alpha=0.8)
    ax2.set_ylabel('异常分数', color='green')
    ax2.tick_params(axis='y')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Transformer和原始数据比较')
    plt.tight_layout()
    plt.savefig(pic_name + ".png")

def plot_anomaly_comparison2(true_data, my_scores, labels, pic_name):
    timesteps = np.arange(len(true_data))

    fig, ax1 = plt.subplots(figsize=(15, 6))

    # 绘制真实数据曲线
    ax1.plot(timesteps, true_data, label='Raw Data', color='black', alpha=0.8)
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('原始值', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

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
    smoothed_scores = moving_average(my_scores, 2)
    timesteps_smooth = timesteps[:len(smoothed_scores)]
    ax2.plot(timesteps_smooth, smoothed_scores, label='Transformer', color='green', alpha=0.8)
    # ax2.plot(timesteps, transformer_scores, label='Transformer', color='orange', alpha=0.8)
    ax2.set_ylabel('异常分数', color='green')
    ax2.tick_params(axis='y')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Transformer和原始数据比较')
    plt.tight_layout()
    plt.savefig(pic_name + ".png")

def plot_anomaly_comparison3(true_data, my_scores, scores, labels, pic_name):
    timesteps = np.arange(len(true_data))

    fig, ax1 = plt.subplots(figsize=(15, 6))

    # 绘制真实数据曲线
    ax1.plot(timesteps, true_data, label='Raw Data', color='black', alpha=0.8)
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('原始值', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

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
    smoothed_scores = moving_average(my_scores, 3)
    s_scores = moving_average(scores, 3)
    timesteps_smooth = timesteps[:len(smoothed_scores)]
    ax2.plot(timesteps_smooth, smoothed_scores, label='TransformerFFT', color='green', alpha=0.8)
    ax1.plot(timesteps_smooth, s_scores, label='Transformer', color='orange', alpha=0.8)
    ax2.set_ylabel('异常分数', color='green')
    ax2.tick_params(axis='y')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Transformer和原始数据比较')
    plt.tight_layout()
    plt.savefig(pic_name + ".png")


def test():
    # ... [你的原有测试代码] ...

    # 获取必要数据
    init_score = np.load('../data/evaluate/init_score.npy')[15587:15687]
    test_score = np.load('../data/evaluate/test_score.npy')[13500:13600]
    test_label = np.load('../data/evaluate/test_label.npy')[13500:13600]
    true_data = init_score[:, 0]  # 假设取第一个特征作为展示
    my_scores = test_score.reshape(-1)
    labels = test_label.reshape(-1)

    # 需要获取Transformer模型的分数（这里需要你自行实现）
    transformer_scores = []  # 替换为实际获取方法
    normal_data = np.random.uniform(1, 2, size=100)

    # 插入异常数据
    normal_data[30:32] = np.random.uniform(5, 6, size=2)

    # 调用绘图函数
    # plot_anomaly_comparison(true_data, my_scores, labels, "comparsion1")
    plot_anomaly_comparison2(true_data, normal_data, labels, "comparsion2")

if __name__ == '__main__':
    test()