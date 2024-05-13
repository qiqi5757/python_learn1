import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 加载数据
fcnn_preds = np.load("pred_label/FCNN_preds.npy")
fcnn_labels = np.load("pred_label/FCNN_labels.npy")
cnn_preds = np.load("pred_label/CNN_preds.npy")
cnn_labels = np.load("pred_label/CNN_labels.npy")
lstm_preds = np.load("pred_label/LSTM_preds.npy")
lstm_labels = np.load("pred_label/LSTM_labels.npy")

# 定义类别
num_classes = 10
classes = np.arange(num_classes)

# 二值化标签，如果已经二值化，跳过这一步
fcnn_labels = label_binarize(fcnn_labels, classes=classes)
cnn_labels = label_binarize(cnn_labels, classes=classes)
lstm_labels = label_binarize(lstm_labels, classes=classes)


# 绘图函数
def plot_average_roc_prc(labels, preds, model_name):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    precisions = []
    recalls = []
    pr_aucs = []

    # 计算每个类的ROC AUC和PRC
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels[:, i], preds[:, i])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        precision, recall, _ = precision_recall_curve(labels[:, i], preds[:, i])
        pr_auc = average_precision_score(labels[:, i], preds[:, i])
        precisions.append(np.interp(mean_fpr, recall[::-1], precision[::-1]))
        recalls.append(recall)
        pr_aucs.append(pr_auc)

    # ROC曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Mean ROC Curve - {model_name}')
    plt.legend(loc="lower right")

    # PRC曲线
    mean_precision = np.mean(precisions, axis=0)
    mean_pr_auc = np.mean(pr_aucs)
    plt.subplot(1, 2, 2)
    plt.plot(mean_fpr, mean_precision, color='b', label=f'Mean PR (AUC = {mean_pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Mean Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.show()


# 绘制曲线
plot_average_roc_prc(fcnn_labels, fcnn_preds, "FCNN")
plot_average_roc_prc(cnn_labels, cnn_preds, "CNN")
plot_average_roc_prc(lstm_labels, lstm_preds, "LSTM")
print('绘制完成')