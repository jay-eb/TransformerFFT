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
from my_exp.model.TransformerFFT import TransformerFFT
from my_exp.utils.earlystop import EarlyStop
from sklearn.metrics import precision_score, recall_score, f1_score

from my_exp.utils.evaluate import evaluate


# from utils.evaluate import evaluate


class Exp:
    def __init__(self, config):
        self.__dict__.update(config)
        self._get_data()
        self._get_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _get_data(self):
        data = getData(
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
            t=self.t
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.early_stopping = EarlyStop(patience=self.patience, path=self.model_dir + self.dataset + '_model.pkl')
        # 使用交叉熵损失, 监督学习
        self.criterion = nn.CrossEntropyLoss()
        # 训练时不带标签，无监督学习
        # self.criterion = nn.MSELoss(reduction='mean')

    def _process_one_batch(self, batch_data, batch_time, batch_stable, batch_label, train):
        batch_data = batch_data.float().to(self.device)
        batch_time = batch_time.float().to(self.device)
        batch_stable = batch_stable.float().to(self.device)
        batch_label = batch_label.long().to(self.device)

        if train:
            # 趋势， 周期， 时间
            stable = self.model(batch_stable, batch_data, batch_time, self.p)
            stable = stable.permute(0, 2, 1)
            batch_label = batch_label.squeeze(-1)
            loss = self.criterion(stable, batch_label)
            return loss
        else:
            stable = self.model(batch_stable, batch_data, batch_time, 0.00)
            return stable

    def train(self):
        for e in range(self.epochs):
            start = time()

            self.model.train()
            train_loss = []
            for (batch_data, batch_time, batch_stable, batch_label) in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                loss = self._process_one_batch(batch_data, batch_time, batch_stable, batch_label, train=True)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                valid_loss = []
                for (batch_data, batch_time, batch_stable, batch_label) in tqdm(self.valid_loader):
                    loss = self._process_one_batch(batch_data, batch_time, batch_stable, batch_label, train=True)
                    valid_loss.append(loss.item())

            train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)
            end = time()
            print(f'Epoch: {e} || Train Loss: {train_loss:.6f} Valid Loss: {valid_loss:.6f} || Cost: {end - start:.4f}')

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break

        self.model.load_state_dict(torch.load(self.model_dir + self.dataset + '_model.pkl'))

    def test(self):
        # 加载训练好的模型
        self.model.load_state_dict(torch.load(self.model_dir + self.dataset + '_model.pkl'))

        with torch.no_grad():
            self.model.eval()
            test_labels = []
            test_preds = []

            for (batch_data, batch_time, batch_stable, batch_label) in tqdm(self.test_loader):
                # 将数据移动到设备
                batch_data = batch_data.float().to(self.device)
                batch_time = batch_time.float().to(self.device)
                batch_stable = batch_stable.float().to(self.device)
                batch_label = batch_label.long().to(self.device)

                # 模型预测
                stable = self.model(batch_stable, batch_data, batch_time, 0.00)
                probabilities = torch.sigmoid(stable)
                preds = (probabilities > 0.5).long()  # 获取每个样本的预测类别（argmax 取概率最大的类别）

                # 保存真实标签和预测值
                test_labels.append(batch_label.squeeze(-1).detach().cpu().numpy())
                test_preds.append(preds.detach().cpu().numpy())

        # 将所有批次的结果合并
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_preds = np.concatenate(test_preds, axis=0).reshape(-1)

        # 计算 Precision、Recall 和 F1-score
        precision = precision_score(test_labels, test_preds, average='binary')  # 二分类情况
        recall = recall_score(test_labels, test_preds, average='binary')
        f1 = f1_score(test_labels, test_preds, average='binary')

        print("\n=============== Evaluation Results ===============")
        print(f"Precision: {precision:.4f} || Recall: {recall:.4f} || F1-Score: {f1:.4f}")
        print("==================================================\n")