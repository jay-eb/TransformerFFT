# data_preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path='sensor_data.csv'):
    """加载传感器数据"""
    data = pd.read_csv(file_path)
    return data

def handle_missing_values(data):
    """处理缺失值"""
    # 检测缺失值
    missing_counts = data.isnull().sum()
    print("各列缺失值数量：")
    print(missing_counts)

    # 前向填充和后向填充处理缺失值
    data = data.fillna(method='ffill').fillna(method='bfill')
    return data

def detect_and_replace_outliers(data, window_size=50, z_thresh=3):
    """使用滑动窗口的 Z-score 方法检测并替换异常值"""
    data_cleaned = data.copy()
    sensor_columns = data.columns.drop(['Timestamp', 'Anomaly'])

    for col in sensor_columns:
        # 计算滑动均值和标准差
        rolling_mean = data_cleaned[col].rolling(window=window_size, center=True).mean()
        rolling_std = data_cleaned[col].rolling(window=window_size, center=True).std()

        # 计算 Z-score
        z_scores = (data_cleaned[col] - rolling_mean) / rolling_std

        # 找到异常值
        outliers = np.abs(z_scores) > z_thresh

        # 用局部中位数替换异常值
        data_cleaned.loc[outliers, col] = data_cleaned[col].rolling(window=window_size, center=True).median()[outliers]

    return data_cleaned

def standardize_data(data):
    """数据标准化处理"""
    data_standardized = data.copy()
    sensor_columns = data.columns.drop(['Timestamp', 'Anomaly'])

    for col in sensor_columns:
        mean = data_standardized[col].mean()
        std = data_standardized[col].std()
        print(f'Before standardization - {col}: mean={mean}, std={std}')
        if std == 0:
            print(f'Warning: Standard deviation of {col} is zero.')
            data_standardized[col] = 0  # 如果标准差为零，所有值设为零
        else:
            data_standardized[col] = (data_standardized[col] - mean) / std
        mean_post = data_standardized[col].mean()
        std_post = data_standardized[col].std()
        print(f'After standardization - {col}: mean={mean_post}, std={std_post}')

    return data_standardized

def align_time_series(data):
    """时间对齐"""
    data_aligned = data.copy()
    # 假设数据已经对齐，则无需处理
    return data_aligned

def preprocess_data(file_path='sensor_data.csv', save_path='sensor_data_preprocessed.csv'):
    """完整的预处理流程"""
    data = load_data(file_path)

    print("开始处理缺失值...")
    data_no_missing = handle_missing_values(data)

    print("开始检测并替换异常值...")
    data_no_outliers = detect_and_replace_outliers(data_no_missing)

    print("开始数据标准化...")
    data_standardized = standardize_data(data_no_outliers)

    print("开始时间对齐...")
    data_aligned = align_time_series(data_standardized)

    print("数据预处理完成，保存为", save_path)
    data_aligned.to_csv(save_path, index=False)

    return data_aligned

def visualize_preprocessing(preprocessed_data, sensor='Engine RPM'):
    """可视化预处理后的数据"""
    plt.figure(figsize=(12, 6))
    plt.plot(preprocessed_data['Timestamp'].values, preprocessed_data[sensor].values, label='Preprocessed', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Standardized ' + sensor)
    plt.title(f'{sensor} - Preprocessed Data')
    plt.legend()
    plt.show()

def check_preprocessed_data(data):
    """检查预处理后数据的基本统计信息"""
    sensor_columns = data.columns.drop(['Timestamp', 'Anomaly'])
    for col in sensor_columns:
        print(f"{col} - min: {data[col].min()}, max: {data[col].max()}, mean: {data[col].mean()}, std: {data[col].std()}")

def check_data_integrity(data):
    """检查数据完整性"""
    print("Checking for NaN values:")
    print(data.isna().sum())
    print("Checking for infinite values:")
    print(np.isinf(data).sum())

def check_variance_before_standardization(data):
    """检查标准化前的数据方差"""
    sensor_columns = data.columns.drop(['Timestamp', 'Anomaly'])
    for col in sensor_columns:
        variance = data[col].var()
        print(f"{col} - Variance before standardization: {variance}")

def main():
    original_data = load_data()
    print("检查标准化前的数据方差:")
    check_variance_before_standardization(original_data)

    preprocessed_data = preprocess_data()

    print("检查预处理后数据的基本统计信息:")
    check_preprocessed_data(preprocessed_data)

    print("检查预处理后数据的完整性:")
    check_data_integrity(preprocessed_data)

    # 仅可视化预处理后的数据
    visualize_preprocessing(preprocessed_data, sensor='Engine RPM')
    visualize_preprocessing(preprocessed_data, sensor='Coolant Temperature')
    visualize_preprocessing(preprocessed_data, sensor='Oil Pressure')

if __name__ == "__main__":
    main()

