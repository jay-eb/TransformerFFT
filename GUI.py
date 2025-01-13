import sys
import os
import random
from datetime import datetime
import logging

import psutil
import numpy as np
import torch
import torch.nn as nn

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QProgressBar, QTextEdit, QMessageBox, QDialog
)

import pyqtgraph as pg

# 设置日志记录
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义模型架构
class HybridModel(nn.Module):
    """混合模型（LSTM + Transformer，适用于特征数据的版本）"""
    def __init__(self, input_size=5):  # 修改 input_size 为 5
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

class RealTimePlot(QWidget):
    def __init__(self, parent=None):
        super(RealTimePlot, self).__init__(parent)

        # 创建一个图表部件
        self.plot_widget = pg.PlotWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        # 设置曲线
        self.curve1 = self.plot_widget.plot(pen=pg.mkPen(color='#3498db', width=2), name='传感器1')
        self.curve2 = self.plot_widget.plot(pen=pg.mkPen(color='#e74c3c', width=2), name='传感器2')

        # 数据列表
        self.xdata = []
        self.ydata1 = []
        self.ydata2 = []

        # 设置图表属性
        self.plot_widget.setTitle('实时监控数据', color='#2c3e50', size='12pt')
        self.plot_widget.setLabel('left', '传感器值')
        self.plot_widget.setLabel('bottom', '时间')
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)

    @pyqtSlot(str, float, float)
    def update_plot(self, new_time, new_val1, new_val2):
        self.xdata.append(new_time)
        self.ydata1.append(new_val1)
        self.ydata2.append(new_val2)

        self.xdata = self.xdata[-100:]
        self.ydata1 = self.ydata1[-100:]
        self.ydata2 = self.ydata2[-100:]

        x_indexes = list(range(len(self.xdata)))

        self.curve1.setData(x_indexes, self.ydata1)
        self.curve2.setData(x_indexes, self.ydata2)

        self.plot_widget.setXRange(max(0, len(self.xdata) - 100), len(self.xdata))

class WorkerSignals(QObject):
    log = pyqtSignal(str)
    update_anomaly = pyqtSignal(int, list)
    update_plot = pyqtSignal(str, float, float)

class DetectionWorker(QThread):
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.signals = WorkerSignals()
        self.running = True
        self.data = data
        self.index = 0  # 用于遍历数据集

    def run(self):
        try:
            features = self.data['features']
            labels = self.data['labels']
            num_samples = features.shape[0]

            while self.running:
                feature = features[self.index]
                input_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # [batch_size, input_size]

                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = torch.softmax(output, dim=1).numpy()[0]
                    prediction = np.argmax(probabilities)

                self.signals.update_anomaly.emit(prediction, probabilities.tolist())

                sensor_val1 = feature[0]
                sensor_val2 = feature[1] if feature.shape[0] > 1 else 0  # 防止索引越界
                current_time = datetime.now().strftime("%H:%M:%S")
                self.signals.update_plot.emit(current_time, sensor_val1, sensor_val2)

                if prediction == 0:
                    log_msg = f"检测完成: 未发现异常"
                elif prediction == 1:
                    log_msg = f"检测完成: 发现异常 (传感器 ID: {random.randint(1000,9999)}) - 已发送警报"
                self.signals.log.emit(log_msg)

                self.index += 1
                if self.index >= num_samples:
                    self.index = 0  # 如果需要循环，可以重置索引

                self.msleep(100)  # 线程休眠100毫秒

            self.signals.log.emit("检测任务已停止。")

        except Exception as e:
            error_msg = f"检测过程中发生错误: {str(e)}"
            self.signals.log.emit(error_msg)

    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能汽车异常检测系统")
        self.setGeometry(100, 100, 1600, 1000)  # 将窗口尺寸调整为更大

        self.setStyleSheet("background-color: #ecf0f1;")

        # 设置全局字体
        font = QFont("SimHei", 12)
        self.setFont(font)

        self.model = None
        self.detection_worker = None
        self.detection_running = False
        self.data = None  # 用于存储加载的数据

        # 模型和数据文件路径（请根据实际路径修改）
        self.default_model_path = r"path_to_your_model\Hybrid_best.pth"
        self.default_data_path = r"path_to_your_data\features_labels.npz"

        # 初始化计数器
        self.total_predictions = 0
        self.normal_count = 0
        self.abnormal_count = 0

        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        header = self.create_header()
        main_layout.addWidget(header)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        sidebar = self.create_sidebar()
        content_layout.addWidget(sidebar)

        main_content = self.create_main_content()
        content_layout.addWidget(main_content)

        status_bar = self.create_status_bar()
        main_layout.addWidget(status_bar)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_status)
        self.timer.start(1000)

    def create_header(self):
        header = QWidget()
        header.setFixedHeight(60)
        header.setStyleSheet("""
            background: qlineargradient(x1:0%, y1:0%, x2:100%, y2:0%, 
                                        stop:0% #2c3e50, stop:100% #34495e);
        """)
        layout = QHBoxLayout()
        header.setLayout(layout)

        title = QLabel("智能汽车异常检测系统")
        title.setFont(QFont("SimHei", 24))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title, alignment=Qt.AlignLeft)

        return header

    def create_sidebar(self):
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("background-color: #34495e;")
        layout = QVBoxLayout()
        sidebar.setLayout(layout)

        control_label = QLabel("控制面板")
        control_label.setFont(QFont("SimHei", 18))
        control_label.setStyleSheet("color: #ecf0f1;")
        control_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(control_label)

        button_style = """
            QPushButton {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
        """

        self.load_model_button = QPushButton("加载模型")
        self.load_model_button.setStyleSheet(button_style)
        self.load_model_button.clicked.connect(self.load_model)
        layout.addWidget(self.load_model_button)

        self.start_button = QPushButton("启动检测")
        self.start_button.setStyleSheet(button_style)
        self.start_button.clicked.connect(self.start_detection)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("停止检测")
        self.stop_button.setStyleSheet(button_style)
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        self.report_button = QPushButton("生成报告")
        self.report_button.setStyleSheet(button_style)
        self.report_button.clicked.connect(self.generate_report)
        self.report_button.setEnabled(False)
        layout.addWidget(self.report_button)

        self.settings_button = QPushButton("系统设置")
        self.settings_button.setStyleSheet(button_style)
        self.settings_button.clicked.connect(self.open_settings)
        layout.addWidget(self.settings_button)

        layout.addStretch()

        return sidebar

    def create_main_content(self):
        main_content = QWidget()
        main_layout = QVBoxLayout()
        main_content.setLayout(main_layout)

        self.plot = RealTimePlot(self)
        main_layout.addWidget(self.plot)

        results_status_layout = QHBoxLayout()
        main_layout.addLayout(results_status_layout)

        anomaly_widget = self.create_anomaly_results()
        results_status_layout.addWidget(anomaly_widget)

        system_status_widget = self.create_system_status()
        results_status_layout.addWidget(system_status_widget)

        logs_widget = self.create_logs()
        main_layout.addWidget(logs_widget)

        return main_content

    def create_anomaly_results(self):
        widget = QWidget()
        widget.setStyleSheet("background-color: #ffffff; border-radius: 10px;")
        widget.setFixedHeight(250)  # 增加高度
        layout = QVBoxLayout()
        widget.setLayout(layout)

        title = QLabel("异常检测结果")
        title.setFont(QFont("SimHei", 16))
        title.setStyleSheet("color: #2c3e50;")
        layout.addWidget(title)

        normal_label = QLabel("正常比例")
        normal_label.setFont(QFont("SimHei", 12))
        layout.addWidget(normal_label)

        self.normal_progress = QProgressBar()
        self.normal_progress.setValue(0)
        self.normal_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #2c3e50;
                border-radius: 5px;
                text-align: center;
                height: 30px;  # 调整高度
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 20px;
            }
        """)
        layout.addWidget(self.normal_progress)

        abnormal_label = QLabel("异常比例")
        abnormal_label.setFont(QFont("SimHei", 12))
        layout.addWidget(abnormal_label)

        self.abnormal_progress = QProgressBar()
        self.abnormal_progress.setValue(0)
        self.abnormal_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #2c3e50;
                border-radius: 5px;
                text-align: center;
                height: 30px;  # 调整高度
            }
            QProgressBar::chunk {
                background-color: #e74c3c;
                width: 20px;
            }
        """)
        layout.addWidget(self.abnormal_progress)

        return widget

    def create_system_status(self):
        widget = QWidget()
        widget.setStyleSheet("background-color: #ffffff; border-radius: 10px;")
        widget.setFixedHeight(250)  # 增加高度
        layout = QVBoxLayout()
        widget.setLayout(layout)

        title = QLabel("系统状态")
        title.setFont(QFont("SimHei", 16))
        title.setStyleSheet("color: #2c3e50;")
        layout.addWidget(title)

        online_layout = QHBoxLayout()
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(20, 20)
        self.status_indicator.setStyleSheet("background-color: #2ecc71; border-radius: 10px;")
        online_layout.addWidget(self.status_indicator)

        self.status_text = QLabel("系统正常运行中")
        self.status_text.setFont(QFont("SimHei", 12))
        online_layout.addWidget(self.status_text)

        layout.addLayout(online_layout)

        self.cpu_label = QLabel("CPU 使用率: 0%")
        self.cpu_label.setFont(QFont("SimHei", 12))
        layout.addWidget(self.cpu_label)

        self.cpu_progress = QProgressBar()
        self.cpu_progress.setValue(0)
        self.cpu_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #2c3e50;
                border-radius: 5px;
                text-align: center;
                height: 30px;  # 调整高度
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 20px;
            }
        """)
        layout.addWidget(self.cpu_progress)

        self.mem_label = QLabel("内存使用率: 0%")
        self.mem_label.setFont(QFont("SimHei", 12))
        layout.addWidget(self.mem_label)

        self.mem_progress = QProgressBar()
        self.mem_progress.setValue(0)
        self.mem_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #2c3e50;
                border-radius: 5px;
                text-align: center;
                height: 30px;  # 调整高度
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 20px;
            }
        """)
        layout.addWidget(self.mem_progress)

        return widget

    def create_logs(self):
        widget = QWidget()
        widget.setStyleSheet("background-color: #ffffff; border-radius: 10px;")
        # 增加日志窗口高度
        widget.setFixedHeight(300)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        title = QLabel("最近检测日志")
        title.setFont(QFont("SimHei", 16))
        title.setStyleSheet("color: #2c3e50;")
        layout.addWidget(title)

        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setStyleSheet("""
            QTextEdit {
                background-color: #ecf0f1;
                border: none;
                font-family: 'SimHei', sans-serif;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.logs_text)

        return widget

    def create_status_bar(self):
        status_bar = QWidget()
        status_bar.setFixedHeight(40)
        status_bar.setStyleSheet("background-color: #34495e;")
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 0, 20, 0)
        status_bar.setLayout(layout)

        self.time_label = QLabel("当前时间: 2024-10-15 12:00:00")
        self.time_label.setFont(QFont("SimHei", 12))
        self.time_label.setStyleSheet("color: #ecf0f1;")
        layout.addWidget(self.time_label, alignment=Qt.AlignLeft)

        online_status = QLabel()
        online_status.setFixedSize(20, 20)
        online_status.setStyleSheet("background-color: #2ecc71; border-radius: 10px;")
        layout.addWidget(online_status, alignment=Qt.AlignRight)

        online_text = QLabel("在线")
        online_text.setFont(QFont("SimHei", 12))
        online_text.setStyleSheet("color: #ecf0f1;")
        layout.addWidget(online_text, alignment=Qt.AlignRight)

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_time)
        self.status_timer.start(1000)

        return status_bar

    def update_time(self):
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_time = f"当前时间: {current_time_str}"
        self.time_label.setText(current_time)

    def update_system_status(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent

        self.cpu_progress.setValue(int(cpu))
        self.mem_progress.setValue(int(mem))

        self.cpu_label.setText(f"CPU 使用率: {cpu}%")
        self.mem_label.setText(f"内存使用率: {mem}%")

    def load_model(self):
        # 加载模型
        model_file, _ = QFileDialog.getOpenFileName(self, "加载模型", "", "PTH Files (*.pth)")
        if not model_file:
            model_file = self.default_model_path

        try:
            input_size = 5  # 确保 input_size 与模型匹配

            self.model = HybridModel(input_size)
            self.model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
            self.model.eval()
            self.log(f"模型 {os.path.basename(model_file)} 加载成功。")

            # 加载数据
            data_file, _ = QFileDialog.getOpenFileName(self, "加载数据", "", "NPZ Files (*.npz)")
            if not data_file:
                data_file = self.default_data_path

            self.data = np.load(data_file)
            self.log(f"数据 {os.path.basename(data_file)} 加载成功。")

            # 检查特征维度
            features = self.data['features']
            if features.shape[1] != input_size:
                QMessageBox.critical(self, "错误", f"特征维度 ({features.shape[1]}) 与模型输入大小 ({input_size}) 不匹配。")
                self.model = None
                self.data = None
                return

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型或数据失败: {str(e)}")
            self.model = None
            self.data = None
            return

    def start_detection(self):
        if not self.model or self.data is None:
            QMessageBox.warning(self, "警告", "请先加载模型和数据。")
            return

        if self.detection_running:
            QMessageBox.warning(self, "警告", "检测已经在运行中。")
            return

        # 重置计数器
        self.total_predictions = 0
        self.normal_count = 0
        self.abnormal_count = 0
        self.normal_progress.setValue(0)
        self.abnormal_progress.setValue(0)

        self.detection_running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.report_button.setEnabled(False)
        self.log("检测开始。")

        self.detection_worker = DetectionWorker(self.model, self.data)
        self.detection_worker.signals.log.connect(self.log)
        self.detection_worker.signals.update_anomaly.connect(self.update_anomaly_results)
        self.detection_worker.signals.update_plot.connect(self.update_plot)
        self.detection_worker.start()

    def stop_detection(self):
        if self.detection_worker:
            self.detection_worker.stop()
            self.detection_worker.wait()
            self.detection_worker = None
            self.detection_running = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.report_button.setEnabled(True)
            self.log("检测已停止。")

    def generate_report(self):
        report_content = "智能汽车异常检测报告\n"
        report_content += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report_content += "异常检测结果:\n"
        report_content += f"已检测数据总数: {self.total_predictions}\n"
        report_content += f"正常数据数: {self.normal_count}\n"
        report_content += f"异常数据数: {self.abnormal_count}\n"
        report_content += f"正常比例: {self.normal_progress.value()}%\n"
        report_content += f"异常比例: {self.abnormal_progress.value()}%\n\n"
        report_content += "系统状态:\n"
        report_content += f"CPU 使用率: {self.cpu_progress.value()}%\n"
        report_content += f"内存使用率: {self.mem_progress.value()}%\n"

        report_path, _ = QFileDialog.getSaveFileName(self, "保存报告", "", "Text Files (*.txt)")
        if not report_path:
            return
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.log(f"报告已生成并保存至: {report_path}")
            QMessageBox.information(self, "成功", "报告生成成功。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成报告失败: {str(e)}")

    def open_settings(self):
        settings_dialog = QDialog()
        settings_dialog.setWindowTitle("系统设置")
        settings_dialog.setFixedSize(400, 300)
        layout = QVBoxLayout()
        label = QLabel("系统设置内容在此添加。")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        settings_dialog.setLayout(layout)
        settings_dialog.exec_()

    @pyqtSlot(str)
    def log(self, message):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        self.logs_text.append(f"{timestamp} {message}")

    @pyqtSlot(int, list)
    def update_anomaly_results(self, prediction, probability):
        # 更新计数器
        self.total_predictions += 1
        if prediction == 0:
            self.normal_count += 1
        else:
            self.abnormal_count += 1

        # 计算累计比例
        normal_percentage = (self.normal_count / self.total_predictions) * 100
        abnormal_percentage = (self.abnormal_count / self.total_predictions) * 100

        # 更新进度条
        self.normal_progress.setValue(int(normal_percentage))
        self.abnormal_progress.setValue(int(abnormal_percentage))

    @pyqtSlot(str, float, float)
    def update_plot(self, current_time, val1, val2):
        self.plot.update_plot(current_time, val1, val2)

    def closeEvent(self, event):
        if self.detection_worker and self.detection_worker.isRunning():
            self.detection_worker.stop()
            self.detection_worker.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    # 设置全局字体
    font = QFont("SimHei", 12)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
