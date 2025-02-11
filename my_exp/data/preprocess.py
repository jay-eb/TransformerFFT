import numpy as np
import pandas as pd
from scipy.fftpack import irfft, rfft
import torch
from sklearn.preprocessing import StandardScaler


def getTimeEmbedding(time):
    df = pd.DataFrame(time, columns=['time'])
    df['time'] = pd.to_datetime(df['time'])

    # 把时间特征提取到[-0.5, 0.5] 中心化，中心化方便激活函数计算，也方便捕捉周期性
    df['minute'] = df['time'].apply(lambda row: row.minute / 59 - 0.5)
    df['hour'] = df['time'].apply(lambda row: row.hour / 23 - 0.5)
    df['weekday'] = df['time'].apply(lambda row: row.weekday() / 6 - 0.5)
    df['day'] = df['time'].apply(lambda row: row.day / 30 - 0.5)
    df['month'] = df['time'].apply(lambda row: row.month / 365 - 0.5)

    return df[['minute', 'hour', 'weekday', 'day', 'month']].values

# 滑动平均取趋势
def getStable(data, w=1440):
    trend = pd.DataFrame(data).rolling(w, center=True).median().values
    stable = data - trend
    return data[w // 2:-w // 2, :], stable[w // 2:-w // 2, :]

# Torch版本 FFT获取周期与趋势
def getFFT(data, topK=3):
    tensor = torch.from_numpy(data)
    x_season = torch.zeros_like(tensor)
    x_trend = torch.zeros_like(tensor)
    for i in range(tensor.shape[1]):  # 对每一元数据（每一列）
        x = tensor[:, i]
        # 对该列数据应用傅里叶变换
        xf = torch.fft.rfft(x)
        # 计算频率的幅度
        freq = abs(xf)
        # 设定频率阈值，保留频率较大的部分
        freq[0] = 0  # 保证去除直流分量
        top_k_freq, top_list = torch.topk(freq, topK)
        # 只保留前 top_k 个频率
        xf[freq <= top_k_freq.min()] = 0
        # 通过逆傅里叶变换得到季节性部分
        x_season[:, i] = torch.fft.irfft(xf)
        # 趋势部分即原数据减去季节性
        x_trend[:, i] = x - x_season[:, i]
    return x_season.numpy(), x_trend.numpy()

# Numpy版本 FFT获取周期与趋势
def getNdFFT(data, topK=3):
    x_season = np.zeros_like(data)
    x_trend = np.zeros_like(data)

    # 对每一列数据进行傅里叶变换，得到季节性与趋势性分量
    for i in range(data.shape[1]):  # 对每一元数据（每一列）
        x = data[:, i]
        xf = rfft(x)
        freq = np.abs(xf)
        freq[0] = 0  # 保证去除直流分量
        top_k_freq = np.partition(freq, -topK)[-topK:]  # 获取最大的 top_k 个频率
        top_k_freq_min = top_k_freq.min()  # 计算 top_k 中最小的频率幅度
        xf[freq <= top_k_freq_min] = 0
        x_season[:, i] = irfft(xf)
        x_trend[:, i] = x - x_season[:, i]
    return x_season, x_trend

def getData2(path='./dataset/', dataset='PSM', topK=5, train_rate=0.8, split_test_rate=0.2):
    init_data = np.load(path + dataset + '/' + dataset + '_train_data.npy')
    init_time = getTimeEmbedding(np.load(path + dataset + '/' + dataset + '_train_date.npy'))

    test_data = np.load(path + dataset + '/' + dataset + '_test_data.npy')
    test_time = getTimeEmbedding(np.load(path + dataset + '/' + dataset + '_test_date.npy'))
    test_label = np.load(path + dataset + '/' + dataset + '_test_label.npy')
    # 标准化处理，使得每个特征的均值为0，标准差为1
    scaler = StandardScaler()
    scaler.fit(init_data)
    init_data = pd.DataFrame(scaler.transform(init_data)).fillna(0).values
    test_data = pd.DataFrame(scaler.transform(test_data)).fillna(0).values

    init_data, init_stable = getNdFFT(init_data, topK=topK)
    init_label = np.zeros((len(init_data), 1))
    test_stable = np.zeros_like(test_data)

    train_data = init_data[:int(train_rate * len(init_data)), :]
    train_time = init_time[:int(train_rate * len(init_time)), :]
    train_stable = init_stable[:int(train_rate * len(init_stable)), :]
    train_label = init_label[:int(train_rate * len(init_label)), :]

    valid_data = init_data[int(train_rate * len(init_data)):, :]
    valid_time = init_time[int(train_rate * len(init_time)):, :]
    valid_stable = init_stable[int(train_rate * len(init_stable)):, :]
    valid_label = init_label[int(train_rate * len(init_label)):, :]

    data = {
        'train_data': train_data, 'train_time': train_time, 'train_stable': train_stable, 'train_label': train_label,
        'valid_data': valid_data, 'valid_time': valid_time, 'valid_stable': valid_stable, 'valid_label': valid_label,
        'init_data': init_data, 'init_time': init_time, 'init_stable': init_stable, 'init_label': init_label,
        'test_data': test_data, 'test_time': test_time, 'test_stable': test_stable, 'test_label': test_label
    }

    return data

def getData(path='./dataset/', dataset='PSM', topK=5, train_rate=0.8, split_test_rate=0.2):
    # init_data = np.load(path + dataset + '/' + dataset + '_train_data.npy')
    # init_time = getTimeEmbedding(np.load(path + dataset + '/' + dataset + '_train_date.npy'))

    origin_data = np.load(path + dataset + '/' + dataset + '_test_data.npy')
    origin_time = getTimeEmbedding(np.load(path + dataset + '/' + dataset + '_test_date.npy'))
    origin_label = np.load(path + dataset + '/' + dataset + '_test_label.npy')
    num_train_samples = int(len(origin_data) * (1 - split_test_rate))
    init_data = origin_data[:num_train_samples]
    init_time = origin_time[:num_train_samples]
    init_label = origin_label[:num_train_samples]
    test_data = origin_data[num_train_samples:]
    test_time = origin_time[num_train_samples:]
    test_label = origin_label[num_train_samples:]
    # 标准化处理，使得每个特征的均值为0，标准差为1
    scaler = StandardScaler()
    scaler.fit(init_data)
    init_data = pd.DataFrame(scaler.transform(init_data)).fillna(0).values # 标准化完可能有NaN的值，空值导致的模型训练失败
    test_data = pd.DataFrame(scaler.transform(test_data)).fillna(0).values

    init_data, init_stable = getNdFFT(init_data, topK=topK)
    # 如果是滑动平均，要对时间裁剪 init_time = init_time[period // 2:-period // 2, :]
    # init_label = np.zeros((len(init_data), 1)) # 初始化标签，都是0，即正常的
    test_stable = np.zeros_like(test_data)

    # # 从 test 数据集中抽取一部分作为 train 数据集
    # num_train_samples = int(len(test_data) * (1 - split_test_rate))
    # new_train_data = test_data[:num_train_samples]
    # new_train_time = test_time[:num_train_samples]
    # new_train_label = test_label[:num_train_samples]
    # new_train_stable = test_stable[:num_train_samples]

    # 数据集拆分训练集和验证集
    train_data = init_data[:int(train_rate * len(init_data)), :]
    train_time = init_time[:int(train_rate * len(init_time)), :]
    train_stable = init_stable[:int(train_rate * len(init_stable)), :]
    train_label = init_label[:int(train_rate * len(init_label)), :]

    valid_data = init_data[int(train_rate * len(init_data)):, :]
    valid_time = init_time[int(train_rate * len(init_time)):, :]
    valid_stable = init_stable[int(train_rate * len(init_stable)):, :]
    valid_label = init_label[int(train_rate * len(init_label)):, :]

    data = {
        'train_data': train_data, 'train_time': train_time, 'train_stable': train_stable, 'train_label': train_label,
        'valid_data': valid_data, 'valid_time': valid_time, 'valid_stable': valid_stable, 'valid_label': valid_label,
        'init_data': init_data, 'init_time': init_time, 'init_stable': init_stable, 'init_label': init_label,
        'test_data': test_data, 'test_time': test_time, 'test_stable': test_stable, 'test_label': test_label
    }

    return data

if __name__ == '__main__':
    getData()