from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

matplotlib.use('TkAgg')
# 计算 Precision-Recall 曲线
test_labels = [0]
test_preds = [1]
precision, recall, _ = precision_recall_curve(test_labels, test_preds)
precision1 = precision_score(test_labels, test_preds)
recall_score1 = precision_score(test_labels, test_preds)
f1_score1 = f1_score(test_labels, test_preds)

# 可视化
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("precision_recall_curve.png")
