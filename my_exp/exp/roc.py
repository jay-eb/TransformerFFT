from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# 计算 Precision-Recall 曲线
test_labels = [0]
test_preds = [1]
precision, recall, _ = precision_recall_curve(test_labels, test_preds)

# 可视化
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
