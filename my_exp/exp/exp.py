import os
from time import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from my_exp.data.dataset import Dataset
from my_exp.data.preprocess import getData, getData2
from my_exp.exp.comparison import plot_anomaly_comparison
from my_exp.model.TransformerFFT import TransformerFFT
from my_exp.utils.earlystop import EarlyStop
from sklearn.metrics import precision_score, recall_score, f1_score

from my_exp.utils.evaluate import evaluate
import matplotlib
import matplotlib.pyplot as plt


# from utils.evaluate import evaluate


class Exp:
    def __init__(self, config):
        self.__dict__.update(config)
        self._get_data()
        self._get_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _get_data(self):
        data = getData2(
            path=self.data_dir,
            dataset=self.dataset,
            topK=self.topK,
            train_rate=self.train_rate
        )

        self.feature_num = data['train_data'].shape[1]
        self.time_num = data['train_time'].shape[1]
        print('\ndata shape: ')
        for k, v in data.items():
            print(k, ': ', v.shape)

        self.train_set = Dataset(
            data=data['train_data'],
            time=data['train_time'],
            stable=data['train_stable'],
            label=data['train_label'],
            window_size=self.window_size
        )
        self.valid_set = Dataset(
            data=data['valid_data'],
            time=data['valid_time'],
            stable=data['valid_stable'],
            label=data['valid_label'],
            window_size=self.window_size
        )
        self.init_set = Dataset(
            data=data['init_data'],
            time=data['init_time'],
            stable=data['init_stable'],
            label=data['init_label'],
            window_size=self.window_size
        )
        self.test_set = Dataset(
            data=data['test_data'],
            time=data['test_time'],
            stable=data['test_stable'],
            label=data['test_label'],
            window_size=self.window_size
        )

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.init_loader = DataLoader(self.init_set, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def _get_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\ndevice:', self.device)

        self.model = TransformerFFT(
            time_steps=self.time_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            window_size=self.window_size,
            model_dim=self.model_dim,
            ff_dim=self.ff_dim,
            atten_dim=self.atten_dim,
            feature_num=self.feature_num,
            time_num=self.time_num,
            block_num=self.block_num,
            head_num=self.head_num,
            dropout=self.dropout,
            device=self.device,
            d=self.d,
            t=self.t,
            act=self.act
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.early_stopping = EarlyStop(patience=self.patience, path=self.model_dir + self.dataset + '_model.pkl')
        # 使用交叉熵损失, 监督学习
        # self.criterion = nn.CrossEntropyLoss()
        # 训练时不带标签，无监督学习
        self.criterion = nn.MSELoss(reduction='mean')

    def _process_one_batch(self, batch_data, batch_time, batch_stable, train):
        batch_data = batch_data.float().to(self.device)
        batch_time = batch_time.float().to(self.device)
        batch_stable = batch_stable.float().to(self.device)

        if train:
            # 趋势， 周期， 时间
            trend, period = self.model(batch_stable, batch_data, batch_time, self.p)
            # stable = stable.permute(0, 2, 1)
            # batch_label = batch_label.squeeze(-1)
            # loss = self.criterion(stable, batch_label)
            loss = 0.5 * self.criterion(trend, batch_stable) + \
                   0.5 * self.criterion(period, batch_data)
            return loss
        else:
            trend, period = self.model(batch_stable, batch_data, batch_time, 0.00)
            return trend + period

    def count_parameters(self):
        # Count the total number of parameters in the model
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total number of parameters: {total_params}')
        return total_params

    def train(self):
        epoch_losses = []
        self.count_parameters()
        for e in range(self.epochs):
            start = time()

            self.model.train()
            train_loss = []
            for (batch_data, batch_time, batch_stable, batch_label) in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                loss = self._process_one_batch(batch_data, batch_time, batch_stable, train=True)
                train_loss.append(loss.item())
                epoch_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                valid_loss = []
                for (batch_data, batch_time, batch_stable, batch_label) in tqdm(self.valid_loader):
                    loss = self._process_one_batch(batch_data, batch_time, batch_stable, train=True)
                    valid_loss.append(loss.item())

            train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)
            end = time()
            print(f'Epoch: {e} || Train Loss: {train_loss:.6f} Valid Loss: {valid_loss:.6f} || Cost: {end - start:.4f}')

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break
        self.model.load_state_dict(torch.load(self.model_dir + self.dataset + '_model.pkl'))
        return epoch_losses

    # def test(self):
    #     # 加载训练好的模型
    #     self.model.load_state_dict(torch.load(self.model_dir + self.dataset + '_model.pkl', map_location=torch.device('cpu')))
    #
    #     with torch.no_grad():
    #         self.model.eval()
    #         test_labels = []
    #         test_preds = []
    #
    #         for (batch_data, batch_time, batch_stable, batch_label) in tqdm(self.test_loader):
    #             # 将数据移动到设备
    #             batch_data = batch_data.float().to(self.device)
    #             batch_time = batch_time.float().to(self.device)
    #             batch_stable = batch_stable.float().to(self.device)
    #             batch_label = batch_label.long().to(self.device)
    #
    #             # 模型预测
    #             stable = self.model(batch_stable, batch_data, batch_time, 0.00)
    #             probabilities = F.softmax(stable, dim=1)
    #             preds = torch.argmax(probabilities, dim=2)  # 获取每个样本的预测类别（argmax 取概率最大的类别）
    #
    #             # 保存真实标签和预测值
    #             test_labels.append(batch_label.squeeze(-1).detach().cpu().numpy())
    #             test_preds.append(preds.detach().cpu().numpy())
    #
    #     # 将所有批次的结果合并
    #     test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    #     test_preds = np.concatenate(test_preds, axis=0).reshape(-1)
    #
    #     # 计算 Precision、Recall 和 F1-score
    #     precision = precision_score(test_labels, test_preds, average='binary')  # 二分类情况
    #     recall = recall_score(test_labels, test_preds, average='binary')
    #     f1 = f1_score(test_labels, test_preds, average='binary')
    #
    #     print("\n=============== Evaluation Results ===============")
    #     print(f"Precision: {precision:.4f} || Recall: {recall:.4f} || F1-Score: {f1:.4f}")
    #     print("==================================================\n")

    def test(self):
        self.model.load_state_dict(torch.load(self.model_dir + self.dataset + '_model.pkl'))

        with torch.no_grad():
            self.model.eval()
            init_src, init_rec = [], []
            for (batch_data, batch_time, batch_stable, batch_label) in tqdm(self.init_loader):
                recon = self._process_one_batch(batch_data, batch_time, batch_stable, train=False)
                init_src.append(batch_data.detach().cpu().numpy()[:, -1, :])
                init_rec.append(recon.detach().cpu().numpy()[:, -1, :])

            test_label, test_src, test_rec = [], [], []
            for (batch_data, batch_time, batch_stable, batch_label) in tqdm(self.test_loader):
                recon = self._process_one_batch(batch_data, batch_time, batch_stable, train=False)
                test_label.append(batch_label.detach().cpu().numpy()[:, -1, :])
                test_src.append(batch_data.detach().cpu().numpy()[:, -1, :])
                test_rec.append(recon.detach().cpu().numpy()[:, -1, :])

        init_src = np.concatenate(init_src, axis=0)
        init_rec = np.concatenate(init_rec, axis=0)
        init_mse = (init_src - init_rec) ** 2

        test_label = np.concatenate(test_label, axis=0)
        test_src = np.concatenate(test_src, axis=0)
        test_rec = np.concatenate(test_rec, axis=0)
        test_mse = (test_src - test_rec) ** 2

        init_score = np.mean(init_mse, axis=-1, keepdims=True)
        test_score = np.mean(test_mse, axis=-1, keepdims=True)

        # 获取必要数据
        true_data = init_score[:, 0]  # 假设取第一个特征作为展示
        my_scores = test_score.reshape(-1)
        labels = test_label.reshape(-1)
        plot_anomaly_comparison(true_data, my_scores, [], labels)

        res = evaluate(init_score.reshape(-1), test_score.reshape(-1), test_label.reshape(-1), q=self.q)
        print("\n=============== " + self.dataset + " ===============")
        print(f"P: {res['precision']:.4f} || R: {res['recall']:.4f} || F1: {res['f1_score']:.4f}")
        print("=============== " + self.dataset + " ===============\n")

    def plot_anomaly_comparison(true_data, my_scores, transformer_scores, labels):
        matplotlib.use('TkAgg')
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
        plt.savefig("comparison.png")
