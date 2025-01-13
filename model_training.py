# model_training.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path='features_labels.npz'):
    """加载特征和标签"""
    data = np.load(file_path)
    features = data['features']
    labels = data['labels']
    return features, labels

def preprocess_data(features, labels):
    """数据划分和转换"""
    X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)  # 0.25 x 0.8 = 0.2
    print(f"训练集样本数：{len(X_train)}")
    print(f"验证集样本数：{len(X_val)}")
    print(f"测试集样本数：{len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

class SensorDataset(Dataset):
    """自定义数据集"""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class LSTMModel(nn.Module):
    """LSTM 模型（适用于特征数据的版本）"""
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, 16)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(16, 2)
    def forward(self, x):
        # 将输入扩展为 [batch_size, seq_len, input_size]
        x = x.unsqueeze(1)  # seq_len=1
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class TransformerModel(nn.Module):
    """Transformer 模型（适用于特征数据的版本）"""
    def __init__(self, input_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, dropout=0.4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        # 将输入扩展为 [seq_len, batch_size, input_size]
        x = x.unsqueeze(0)  # seq_len=1
        x = self.embedding(x)  # [1, batch_size, 128]
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # [batch_size, 128]
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        return out

class HybridModel(nn.Module):
    """混合模型（LSTM + Transformer，适用于特征数据的版本）"""
    def __init__(self, input_size):
        super(HybridModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dropout=0.4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 2)
    def forward(self, x):
        # 将输入扩展为 [batch_size, seq_len, input_size]
        x = x.unsqueeze(1)  # seq_len=1
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        x = self.transformer_encoder(lstm_out)
        x = x.mean(dim=0)  # [batch_size, hidden_size]
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        return out

class MLPModel(nn.Module):
    """全连接神经网络模型"""
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_name, patience=5):
    """训练模型，包含早停策略"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []
    early_stop_counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # [batch_size, input_size]
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        # 保存最佳模型
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), f'{model_name}_best.pth')
            early_stop_counter = 0  # 重置计数器
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Validation loss did not improve for {patience} epochs. Early stopping.")
                break
    # 加载最佳模型
    model.load_state_dict(torch.load(f'{model_name}_best.pth'))
    return model, train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader, model_name):
    """在测试集上评估模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            scores = nn.functional.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(scores.cpu().numpy())
    # 分类报告
    print(f"{model_name} - Classification Report:")
    print(classification_report(y_true, y_pred))
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    # ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} - ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    return fpr, tpr, roc_auc

def plot_histories(histories, model_names):
    """绘制训练过程中的损失和准确率曲线，比较不同模型"""
    plt.figure(figsize=(20, 6))
    # 绘制验证损失
    plt.subplot(1, 2, 1)
    for i, history in enumerate(histories):
        val_losses = history['val_losses']
        epochs_range = range(len(val_losses))
        plt.plot(epochs_range, val_losses, label=f'{model_names[i]}')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 绘制验证准确率
    plt.subplot(1, 2, 2)
    for i, history in enumerate(histories):
        val_accuracies = history['val_accuracies']
        epochs_range = range(len(val_accuracies))
        plt.plot(epochs_range, val_accuracies, label=f'{model_names[i]}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def compare_roc(models_roc_auc, model_names):
    """比较不同模型的 ROC 曲线"""
    plt.figure(figsize=(8, 6))
    for i, (fpr, tpr, roc_auc) in enumerate(models_roc_auc):
        plt.plot(fpr, tpr, label=f'{model_names[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

def main():
    # 加载数据
    features, labels = load_data()
    print("数据加载完成。")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    input_size = features.shape[1]
    # 数据预处理和划分
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(features, labels)
    # 创建数据集和数据加载器
    batch_size = 64  # 根据需要调整批量大小
    train_dataset = SensorDataset(X_train, y_train)
    val_dataset = SensorDataset(X_val, y_val)
    test_dataset = SensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # 处理类别不平衡，计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"类别权重：{class_weights}")
    # 模型列表
    models = {
        'LSTM': LSTMModel(input_size=input_size),
        'Transformer': TransformerModel(input_size=input_size),
        'Hybrid': HybridModel(input_size=input_size),
        'MLP': MLPModel(input_size=input_size)
    }
    histories = []
    model_names = []
    models_roc_auc = []
    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"开始训练模型：{model_name}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=0.00001)  # 根据需要调整学习率
        num_epochs = 20  # 根据需要调整训练轮数
        model, train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_name, patience=10)
        print(f"{model_name} 模型训练完成。")
        # 保存历史记录
        histories.append({'train_losses': train_losses, 'val_losses': val_losses, 'val_accuracies': val_accuracies})
        model_names.append(model_name)
        # 在测试集上评估模型
        fpr, tpr, roc_auc = evaluate_model(model, test_loader, model_name)
        models_roc_auc.append((fpr, tpr, roc_auc))
        # 保存模型
        torch.save(model.state_dict(), f'{model_name}_model.pth')
        print(f"{model_name} 模型已保存。")
    # 绘制所有模型的验证准确率和损失曲线对比
    plot_histories(histories, model_names)
    # 比较不同模型的 ROC 曲线
    compare_roc(models_roc_auc, model_names)
    print("所有模型的训练和评估已完成。")

if __name__ == "__main__":
    main()
