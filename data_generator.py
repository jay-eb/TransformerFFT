# data_generator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_base_signal(t, b_i, a_i, f_i, signal_type='sin'):
    """生成基础信号，支持不同类型的函数"""
    if signal_type == 'sin':
        return b_i + a_i * np.sin(2 * np.pi * f_i * t)
    elif signal_type == 'cos':
        return b_i + a_i * np.cos(2 * np.pi * f_i * t)
    elif signal_type == 'sawtooth':
        return b_i + a_i * (2 * (t * f_i - np.floor(0.5 + t * f_i)))
    elif signal_type == 'square':
        return b_i + a_i * np.sign(np.sin(2 * np.pi * f_i * t))
    else:
        # 默认使用正弦函数
        return b_i + a_i * np.sin(2 * np.pi * f_i * t)

def generate_random_noise(t, sigma_i):
    """生成随机噪声，包括高斯噪声和均匀噪声"""
    noise_type = np.random.choice(['gaussian', 'uniform'])
    if noise_type == 'gaussian':
        return np.random.normal(0, sigma_i, size=len(t))
    else:
        return np.random.uniform(-sigma_i, sigma_i, size=len(t))

def generate_anomaly(t, anomaly_intervals, anomaly_type, params):
    """生成异常信号"""
    anomaly_signal = np.zeros_like(t)
    for interval in anomaly_intervals:
        start_idx = np.searchsorted(t, interval[0])
        end_idx = np.searchsorted(t, interval[1])
        length = end_idx - start_idx
        if anomaly_type == 'spike':
            # 尖峰异常，随机正负
            amplitude = params['delta_i'] * np.random.uniform(1, 2)
            sign = np.random.choice([-1, 1])
            anomaly_signal[start_idx:end_idx] = sign * amplitude
        elif anomaly_type == 'drift':
            # 漂移异常，随机上升或下降
            drift = np.linspace(0, params['delta_i'] * np.random.choice([-1, 1]), length)
            anomaly_signal[start_idx:end_idx] = drift
        elif anomaly_type == 'noise':
            # 噪声异常，增大噪声幅度
            noise = np.random.normal(0, abs(params['delta_i']) * 2, size=length)
            anomaly_signal[start_idx:end_idx] = noise
        elif anomaly_type == 'drop':
            # 信号突然下降
            anomaly_signal[start_idx:end_idx] = -params['delta_i']
        elif anomaly_type == 'jump':
            # 信号突然上升
            anomaly_signal[start_idx:end_idx] = params['delta_i']
        elif anomaly_type == 'outlier':
            # 离群点异常
            indices = np.random.choice(range(start_idx, end_idx), size=length // 10, replace=False)
            anomaly_signal[indices] = params['delta_i'] * np.random.uniform(-3, 3, size=len(indices))
    return anomaly_signal

def generate_sensor_data(t, params, anomaly_intervals, anomaly_types):
    """生成单个传感器的数据"""
    # 随机选择基础信号类型
    signal_type = np.random.choice(['sin', 'cos', 'sawtooth', 'square'])
    B_i = generate_base_signal(t, params['b_i'], params['a_i'], params['f_i'], signal_type=signal_type)
    P_i = generate_base_signal(t, params['c_i'], params['a_p'], params['f_p'], signal_type='sin')
    N_i = generate_random_noise(t, params['sigma_i'])
    A_i = np.zeros_like(t)
    for anomaly_type in anomaly_types:
        A_i += generate_anomaly(t, anomaly_intervals, anomaly_type, params)
    x_i = B_i + P_i + N_i + A_i
    return x_i

def generate_dataset():
    """生成完整的数据集"""
    total_time = 40000  # 总时间，单位为秒，进一步增加总时间
    sampling_rate = 50  # 采样率，每秒50个数据点，增加采样率
    t = np.linspace(0, total_time, int(total_time * sampling_rate))

    # 定义异常发生的时间区间，增加异常事件数量
    anomaly_intervals = []
    num_anomalies = 1000  # 增加异常数量
    for _ in range(num_anomalies):
        start_time = np.random.uniform(0, total_time - 10)
        duration = np.random.uniform(1, 30)
        anomaly_intervals.append((start_time, start_time + duration))

    # 定义异常类型，增加更多类型
    anomaly_types_list = ['spike', 'drift', 'noise', 'drop', 'jump', 'outlier']

    # 定义每个传感器的参数
    sensor_params = {
        'Engine RPM': {'b_i': 2000, 'a_i': 500, 'f_i': 0.0005, 'c_i': 100, 'a_p': 50, 'f_p': 0.005, 'sigma_i': 50, 'delta_i': 1500},
        'Vehicle Speed': {'b_i': 80, 'a_i': 20, 'f_i': 0.0002, 'c_i': 5, 'a_p': 2, 'f_p': 0.002, 'sigma_i': 5, 'delta_i': 50},
        'Coolant Temperature': {'b_i': 90, 'a_i': 5, 'f_i': 0.0001, 'c_i': 2, 'a_p': 1, 'f_p': 0.001, 'sigma_i': 1, 'delta_i': 20},
        'Oil Pressure': {'b_i': 50, 'a_i': 10, 'f_i': 0.0003, 'c_i': 3, 'a_p': 1.5, 'f_p': 0.003, 'sigma_i': 2, 'delta_i': 30},
        'Fuel Consumption Rate': {'b_i': 8, 'a_i': 2, 'f_i': 0.0002, 'c_i': 0.5, 'a_p': 0.2, 'f_p': 0.0015, 'sigma_i': 0.2, 'delta_i': 5},
        'Battery Voltage': {'b_i': 14, 'a_i': 0.5, 'f_i': 0.0003, 'c_i': 0.1, 'a_p': 0.05, 'f_p': 0.0025, 'sigma_i': 0.05, 'delta_i': 2},
        'Intake Air Temperature': {'b_i': 25, 'a_i': 3, 'f_i': 0.00015, 'c_i': 1, 'a_p': 0.5, 'f_p': 0.0012, 'sigma_i': 0.5, 'delta_i': 10},
        'Exhaust Temperature': {'b_i': 300, 'a_i': 50, 'f_i': 0.0002, 'c_i': 10, 'a_p': 5, 'f_p': 0.002, 'sigma_i': 5, 'delta_i': 100},
        'Transmission Temperature': {'b_i': 80, 'a_i': 5, 'f_i': 0.0002, 'c_i': 2, 'a_p': 1, 'f_p': 0.0018, 'sigma_i': 1, 'delta_i': 20},
        'Ambient Temperature': {'b_i': 20, 'a_i': 10, 'f_i': 0.00005, 'c_i': 5, 'a_p': 2, 'f_p': 0.0005, 'sigma_i': 0.5, 'delta_i': 15},
        'Tire Pressure': {'b_i': 35, 'a_i': 2, 'f_i': 0.00015, 'c_i': 0.5, 'a_p': 0.2, 'f_p': 0.001, 'sigma_i': 0.2, 'delta_i': 5},
    }

    data = pd.DataFrame({'Timestamp': t})

    # 初始化标签列，0表示正常，1表示异常
    data['Anomaly'] = 0

    for sensor_name, params in sensor_params.items():
        # 对于每个传感器，随机选择异常类型
        sensor_anomaly_types = np.random.choice(anomaly_types_list, size=len(anomaly_intervals), replace=True)
        sensor_data = generate_sensor_data(t, params, anomaly_intervals, sensor_anomaly_types)
        data[sensor_name] = sensor_data

    # 更新标签列
    for interval in anomaly_intervals:
        start_idx = np.searchsorted(t, interval[0])
        end_idx = np.searchsorted(t, interval[1])
        data.loc[start_idx:end_idx, 'Anomaly'] = 1

    return data

def main():
    data = generate_dataset()
    data.to_csv('sensor_data.csv', index=False)
    print("数据生成完成，保存为 sensor_data.csv")

    # 随机选择部分传感器进行绘制
    selected_sensors = np.random.choice(list(data.columns[1:-1]), size=4, replace=False)

    plt.figure(figsize=(15, 12))

    for i, sensor_name in enumerate(selected_sensors):
        plt.subplot(5, 1, i+1)
        plt.plot(data['Timestamp'], data[sensor_name], label=sensor_name)
        plt.ylabel(sensor_name)
        plt.legend()

    # 绘制标签
    plt.subplot(5, 1, 5)
    plt.plot(data['Timestamp'], data['Anomaly'], label='Anomaly', color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Anomaly')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()