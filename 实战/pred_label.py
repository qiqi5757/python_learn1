# 保存三种模型预测值和真实值
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch

# 检查是否有可用的GPU，如果有，使用GPU；如果没有，使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# Training function
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
            running_loss += loss.item() * inputs.size(0)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader.dataset):.4f}")


def evaluate_and_store(model, test_loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 将列表转换为 numpy 数组
    all_probs = np.vstack(all_probs)
    all_labels = np.hstack(all_labels)

    # 标签二值化
    all_labels = label_binarize(all_labels, classes=range(10))

    return all_probs, all_labels
# 创建模型
fcnn_model = FCNN().to(device)
cnn_model = CNN().to(device)
lstm_model = LSTM().to(device)

# 假设每个模型都已经训练并准备好评估
fcnn_probs, fcnn_labels = evaluate_and_store(fcnn_model, test_loader)
cnn_probs, cnn_labels = evaluate_and_store(cnn_model, test_loader)
lstm_probs, lstm_labels = evaluate_and_store(lstm_model, test_loader)

# 保存到.npy文件
np.save("pred_label/FCNN_preds.npy", fcnn_probs)
np.save("pred_label/FCNN_labels.npy", fcnn_labels)
np.save("pred_label/CNN_preds.npy", cnn_probs)
np.save("pred_label/CNN_labels.npy", cnn_labels)
np.save("pred_label/LSTM_preds.npy", lstm_probs)
np.save("pred_label/LSTM_labels.npy", lstm_labels)

print('保存成功')