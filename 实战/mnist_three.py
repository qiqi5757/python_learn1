# 这是用FCNN（全连接层），CNN，LSTM三个模型来预测手写数字的代码，并绘制出相关的图
# 但是需要注意是ROC和PRC只适合二分类的任务，对于画一个多分类，他只能每个类都给出对应的曲线，非常的繁琐，不够清晰明了
# 当然这作为一个入门的代码已经足够了
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import roc_curve, auc

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

# Calculate ROC curves and AUC for each class using one-vs-rest strategy
def calculate_roc_auc_one_vs_rest(model, test_loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probabilities = nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probabilities.numpy())
            all_labels.extend(labels.numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):  # 10 classes in MNIST
        fpr[i], tpr[i], _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc

# Create and train FCNN model
fcnn_model = FCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fcnn_model.parameters(), lr=0.001)
train(fcnn_model, criterion, optimizer, train_loader)

# Create and train CNN model
cnn_model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
train(cnn_model, criterion, optimizer, train_loader)

# Create and train LSTM model
lstm_model = LSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
train(lstm_model, criterion, optimizer, train_loader)

# Evaluate models
fcnn_preds, fcnn_labels = evaluate(fcnn_model, test_loader)
cnn_preds, cnn_labels = evaluate(cnn_model, test_loader)
lstm_preds, lstm_labels = evaluate(lstm_model, test_loader)

# Calculate ROC curves and AUC for each model using one-vs-rest strategy
fpr_fcnn, tpr_fcnn, roc_auc_fcnn = calculate_roc_auc_one_vs_rest(fcnn_model, test_loader)
fpr_cnn, tpr_cnn, roc_auc_cnn = calculate_roc_auc_one_vs_rest(cnn_model, test_loader)
fpr_lstm, tpr_lstm, roc_auc_lstm = calculate_roc_auc_one_vs_rest(lstm_model, test_loader)

# Plot ROC curves for each model
plt.figure(figsize=(8, 6))
for i in range(10):
    plt.plot(fpr_fcnn[i], tpr_fcnn[i], label=f'FCNN Class {i} (AUC = {roc_auc_fcnn[i]:.2f})')
    plt.plot(fpr_cnn[i], tpr_cnn[i], label=f'CNN Class {i} (AUC = {roc_auc_cnn[i]:.2f})')
    plt.plot(fpr_lstm[i], tpr_lstm[i], label=f'LSTM Class {i} (AUC = {roc_auc_lstm[i]:.2f})')

# plot是绘制二维图表的基础函数
# 横纵坐标都是[0,1]，颜色是深蓝色，lw=linewidth = 2，线条的形状是虚线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for each class using one-vs-rest strategy')
plt.legend()
plt.show()
