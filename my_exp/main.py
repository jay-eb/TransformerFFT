import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib
from scipy.signal import savgol_filter

from exp.exp import Exp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='SMD', help='dataset')
    parser.add_argument('--data_dir', type=str, default='./data/dataset/', help='path of the data')
    parser.add_argument('--model_dir', type=str, default='./checkpoint/', help='path of the checkpoint')

    parser.add_argument('--itr', type=int, default=1, help='num of evaluation')
    parser.add_argument('--epochs', type=int, default=8, help='epoch of train')
    parser.add_argument('--patience', type=int, default=1, help='patience of early stopping')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of data')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of optimizer')

    parser.add_argument('--topK', type=int, default=8, help='FFT变换的topK个频率')
    parser.add_argument('--train_rate', type=float, default=0.8, help='rate of train set')
    parser.add_argument('--window_size', type=int, default=64, help='size of sliding window')

    parser.add_argument('--model_dim', type=int, default=512, help='dimension of hidden layer')
    parser.add_argument('--ff_dim', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--atten_dim', type=int, default=64, help='dimension of various attention')

    parser.add_argument('--block_num', type=int, default=2, help='num of various block')
    parser.add_argument('--head_num', type=int, default=8, help='num of attention head')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout')

    parser.add_argument('--time_steps', type=int, default=1000, help='time step of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='start of diffusion beta')
    parser.add_argument('--beta_end', type=float, default=0.02, help='end of diffusion beta')

    parser.add_argument('--t', type=int, default=500, help='time step of adding noise')
    parser.add_argument('--p', type=float, default=10.00, help='peak value of trend disturbance')
    parser.add_argument('--d', type=int, default=30, help='shift of period')

    parser.add_argument('--q', type=float, default=0.01, help='init anomaly probability of spot')

    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='device ids of gpus')
    parser.add_argument('--act', type=str, default='gelu', help='act')

    config = vars(parser.parse_args())
    # 如果有GPU就取消注释
    torch.cuda.set_device(config['gpu_id'])
    loss_history = {}
    activation_functions =  ['gelu','relu','leakyRelu']

    # for ii in activation_functions:
    #     config['act'] = ii
    #     exp = Exp(config)
    #     losses = exp.train()
    #     loss_history[ii] = losses
        # exp.test()
    # 绘制单个loss
    # matplotlib.use("Agg")
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    # matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # plt.figure(figsize=(8, 6))
    #
    # for act in activation_functions:
    #     label = 'Gelu'
    #     if act == 'gelu':
    #         label = 'Gelu'
    #     elif act == 'relu':
    #         label = 'Relu'
    #     elif act == 'leakyRelu':
    #         label = 'Leaky Relu'
    #     y = loss_history[act]  # 获取损失数据
    #     x = np.arange(len(y))  # 生成 x 轴
    #
    #     # 使用 Savitzky-Golay 滤波器平滑数据
    #     window_length = min(51, len(y) - 1 if len(y) % 2 == 0 else len(y))  # 选择合理的窗口大小
    #     y_smooth = savgol_filter(y, window_length=window_length, polyorder=3)
    #
    #     plt.plot(x, y_smooth, label=label)  # 绘制平滑曲线
    #
    # plt.xlabel("迭代次数")
    # plt.ylabel("损失值")
    # plt.title("不同激活函数下训练过程的损失变化")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("loss.png")

    for ii in range(config['itr']):
        exp = Exp(config)
        exp.train()
        exp.test()
