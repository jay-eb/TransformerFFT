# feature_engineering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from sklearn.decomposition import PCA
from tqdm import tqdm  # 用于显示进度条
import seaborn as sns  # 用于绘制热力图

def load_data(file_path='sensor_data_preprocessed.csv'):
    """加载预处理后的数据"""
    data = pd.read_csv(file_path)
    return data

def sliding_window(data, window_size=500, step_size=250):
    """滑动窗口分割"""
    windows = []
    labels = []
    num_samples = data.shape[0]
    print(f"总样本数：{num_samples}")
    for start in tqdm(range(0, num_samples - window_size + 1, step_size)):
        end = start + window_size
        window = data.iloc[start:end]
        windows.append(window)
        # 标签采用窗口内是否存在异常点
        label = 1 if window['Anomaly'].any() else 0
        labels.append(label)
    return windows, labels

def extract_features(windows):
    """从每个窗口中提取特征"""
    feature_list = []

    print("开始提取特征...")
    for idx, window in enumerate(tqdm(windows)):
        features = {}
        sensor_columns = window.columns.drop(['Timestamp', 'Anomaly'])
        for col in sensor_columns:
            data_series = window[col]
            # 将 data_series 转换为 NumPy 数组
            data_array = data_series.values
            # 统计特征
            features[f'{col}_mean'] = data_array.mean()
            features[f'{col}_std'] = data_array.std()
            features[f'{col}_max'] = data_array.max()
            features[f'{col}_min'] = data_array.min()
            features[f'{col}_skew'] = skew(data_array)
            features[f'{col}_kurtosis'] = kurtosis(data_array)
            # 频域特征
            fft_values = np.abs(fft(data_array))
            fft_values = fft_values[:len(fft_values)//2]  # 取一半
            features[f'{col}_fft_mean'] = np.mean(fft_values)
            features[f'{col}_fft_std'] = np.std(fft_values)
            features[f'{col}_fft_max'] = np.max(fft_values)
        feature_list.append(features)
    feature_df = pd.DataFrame(feature_list)
    return feature_df

def feature_selection(feature_df, labels):
    """特征选择和降维"""
    # 添加标签列
    feature_df['Anomaly'] = labels

    # 提取特征列（不包括标签列）
    feature_columns = feature_df.columns.drop('Anomaly')
    features_only = feature_df[feature_columns]

    # 计算相关性矩阵
    print("计算相关性矩阵...")
    corr_matrix = features_only.corr()

    # 绘制热力图
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.show()

    # 去除相关性高的特征（阈值为0.99）
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column].abs() > 0.99)]
    print(f'将要删除的高相关性特征数量：{len(to_drop)}')
    feature_df_reduced = feature_df.drop(columns=to_drop)

    # 分离标签
    labels = feature_df_reduced['Anomaly']
    features = feature_df_reduced.drop(columns=['Anomaly'])

    # PCA降维
    print("开始PCA降维...")
    pca = PCA(n_components=5)  # 指定保留5个主成分
    features_pca = pca.fit_transform(features)
    print(f'PCA后特征维数：{features_pca.shape[1]}')

    # 可视化PCA解释方差
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid()
    plt.show()
    return features_pca, labels

def save_features(features, labels, save_path='features_labels.npz'):
    """保存特征和标签"""
    np.savez(save_path, features=features, labels=labels)
    print(f'特征和标签已保存为 {save_path}')

def main():
    data = load_data()
    print("数据加载完成。")
    windows, labels = sliding_window(data)
    print(f"窗口数量：{len(windows)}")
    feature_df = extract_features(windows)
    print("特征提取完成。")
    features_pca, labels = feature_selection(feature_df, labels)
    print("特征选择和降维完成。")
    save_features(features_pca, labels)
    print("特征工程流程完成。")

if __name__ == "__main__":
    main()
