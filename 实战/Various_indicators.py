# 保存TP、TN、FP、FN这四种值
# 混淆指标
# TP：正确True预测成正样本Positive
# FN：错误False预测成负样本Negative
# FP：错误False预测成正样本Negative
# TN：正确True预测成负样本Positive
# Recall：正确True预测成正样本/原来正样本的个数（TP/TP+FN）
# Precision：正确True预测为正样本个数/预测为正样本的个数（TP/TP+FP）
# F1: 2*(Precision*Recall)/(Precision+Recall)
# Accuracy：(TP+TN)/(TP+FP+FN+TN)
# IOU：TP/(TP+FP+FN)
# MIOU：平均IOU，[TP/(TP+FP+FN)]/n
# TPR = TP/TP+FN
# FPR = FP/FP+TN
# ROC(AUC)曲线： 纵坐标是TPR，横坐标是FPR
# PRC曲线：精确率（precision）和召回率（recall）之间的关系

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from sklearn.metrics import confusion_matrix
from joblib import dump
from joblib import load

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)

# Create DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

# Define the fully connected neural network model
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        # 因为单个图像本来就没有batch_size一说，所以第一个参数就是通道数，第二个和第三个是高度和宽度
        # 输入维度要一维，所以是28*28=784
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 看见view就知道是换形状，从[batch_size,通道，高度，宽度],-1表示自动匹配数量，比如batch_size是100，那么就自动匹配100个样本数据，后面的784就是通道数*高度*宽度
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 输入的通道数是1，输出的通道时是16，卷积核是3，步长是1，paddin是1(在旁边补充一个为0的像素，这样不会导致边缘没有被提取）
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*14*14, 64)
        self.fc2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = x.view(-1, 16*14*14)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Define the LSTM model
# 长短期神经记忆网络，循环神经网络
# 主要是递归，比如我现在能读懂单词，是因为我之前的积累，我并不会忘记我所有之前学过的东西
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.shape[0],28,-1)
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        x = self.softmax(x)
        return x

# Define training function
# criterion损失函数，通常用于评估真实值和预测值之际的差距
def train(model, criterion, optimizer, train_loader, num_epochs=2):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 计算整个训练周期的平均损失
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Define evaluation function
def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return all_preds, all_labels


###########
def evaluate_and_get_confusion_matrix(model, loader):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)  # 将数据转换在GPU上，这样就不会造成设备的累赘
            outputs = model(data)
           # 找到最大值，并返回最大值所对应的索引，维度 0 通常是批次维（batch size），维度 1 是特征或类别维
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(target.cpu().numpy())
    return confusion_matrix(y_true, y_pred)

# 计算每个类的TP, TN, FP, FN
def calculate_metrics(cm):
    # cm 通常指的是混淆矩阵（confusion matrix），cm[2, 3]的值是5，这意味着实际为类别2的样本中有5个被错误地预测为类别3。
    # axis=0通常代表沿着列的方向（垂直方向）
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm) # 提取对角线元素
    TN = cm.sum() - (FP + FN + TP)
    return TP, TN, FP, FN

# 训练和评估模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = [FCNN(), CNN(), LSTM()]
model_names = ['FCNN', 'CNN', 'LSTM']
results = {}

# 指定保存结果的文件夹名称
folder_name = "./result/model_metrics"

# 检查文件夹是否存在，如果不存在，则创建它
if not os.path.exists(folder_name):
    os.makedirs(folder_name,encodings='utf-8')


for model, name in zip(models, model_names):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, criterion, optimizer, train_loader)  # 训练函数需要定义
    cm = evaluate_and_get_confusion_matrix(model, test_loader)
    TP, TN, FP, FN = calculate_metrics(cm)
    results[name] = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

    # 构建保存文件的完整路径
    save_path = os.path.join(folder_name, f'{name}_metrics.joblib')
    # 使用joblib保存结果
    dump(results[name], save_path)

print("Metrics saved using joblib.")

# 文件名列表
file_names = ['FCNN_metrics.joblib', 'CNN_metrics.joblib', 'LSTM_metrics.joblib']

# 循环遍历文件名，加载每个文件
for file_name in file_names:
    file_path = os.path.join(folder_name, file_name)
    data = load(file_path)
    print(f"Data from {file_name}:")
    print(data)
    print("\n")  # 添加换行以便于阅读输出
    print("保存成功")