# 将画三条线，分别是FCNN、CNN、LSTM对模型的预测结果图，图例中只有三条线，分别对应FCNN、CNN、LSTM对于10种类别加和求平均的结果
# AUC曲线：全称AUROC，是曲线下面积，Area Under the ROC Curve，真阳性率（True Positive Rate, TPR）和假阳性率（False Positive Rate, FPR）之间的关系
# PRC曲线：全称平均的精确度-召回率曲线（Precision-Recall Curve, PRC）
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# 数据转换和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)

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
# loss损失值提供了反馈信号用于模型训练，指导权重调整方向，损失值小：表示模型的预测值与真实值非常接近
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


# 新的评估和计算ROC函数
def calculate_average_roc(model, test_loader, num_classes=10):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probabilities = nn.functional.softmax(outputs, dim=1) # 预测概率
            # 存储预测值和真实值
            all_probs.extend(probabilities.numpy()) # 将tensor向量转化成numpy，存储在all_probs列表里
            all_labels.extend(labels.numpy()) # 将labels标签也转换成numpy数组，存储在all_labels列表里

    # 将标签二值化
    all_labels = label_binarize(all_labels, classes=range(num_classes))
    # 进行one-hot编码，比如说all_labels[1,2,3],1=[1,0,0],2=[0,1,0],3=[0,0,1]
    all_probs = np.array(all_probs) # 将列表变成numpy数组，all_probs可能是通过逐个批次累加预测概率而构成的列表，每个元素是一个小批量的概率向量。

    # 计算平均ROC AUC
    mean_fpr = np.linspace(0, 1, 100) # 将0-1分成100份，也就是0.01就取一个值，从而可以绘制出一条曲线
    # 用于存储每次迭代或模型评估计算的结果
    tprs = []
    aucs = []
    for i in range(num_classes):
        # 这行代码是用来计算每个类别（标记为 i）的假正率（False Positive Rate, FPR）和真正率（True Positive Rate, TPR）的
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        # mean_fpr是指在一定范围内的平均fpr，根据给定的mean_fpr,fpr,tpr计算生成一个新的tpr值
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        # 确保曲线从0开始
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    # 设置最后一个tpr值为1，这样可以保证曲线有始有终是在0-1的曲线
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    return mean_fpr, mean_tpr, mean_auc


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

# 绘制模型的平均ROC曲线
def plot_average_roc():
    plt.figure(figsize=(8, 6))

    for model, name in [(fcnn_model, 'FCNN'), (cnn_model, 'CNN'), (lstm_model, 'LSTM')]:
        mean_fpr, mean_tpr, mean_auc = calculate_average_roc(model, test_loader)
        plt.plot(mean_fpr, mean_tpr, label=f'{name} (AUC = {mean_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curve for all classes')
    plt.legend(loc="lower right")  # 将图例放在右下角
    plt.savefig('./images/AUC.png')
    plt.show()
    print('AUC曲线已绘制完成')


def calculate_average_prc(model, test_loader, num_classes=10):
    model.eval() # 进行模型评估，BatchNorm使用训练期间估计的全局统计数据进行归一化，Dropout层不进行任何操作
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probabilities = nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probabilities.numpy())
            all_labels.extend(labels.numpy())

    # 将标签二值化
    all_labels = label_binarize(all_labels, classes=range(num_classes))
    all_probs = np.array(all_probs)

    # 计算平均PRC AUC，只需要进行插值和平均处理的选择
    mean_recall = np.linspace(0, 1, 100)
    # 而precision数值就不需要求均值，因为他会根据recall的值来确定
    precisions = []
    pr_aucs = []
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(interp_precision)
        pr_aucs.append(average_precision_score(all_labels[:, i], all_probs[:, i]))

    mean_precision = np.mean(precisions, axis=0)
    mean_pr_auc = np.mean(pr_aucs)

    return mean_recall, mean_precision, mean_pr_auc


# 绘制模型的平均PRC曲线
def plot_average_prc():
    plt.figure(figsize=(8, 6))

    for model, name in [(fcnn_model, 'FCNN'), (cnn_model, 'CNN'), (lstm_model, 'LSTM')]:
        mean_recall, mean_precision, mean_pr_auc = calculate_average_prc(model, test_loader)
        plt.plot(mean_recall, mean_precision, label=f'{name} (AP = {mean_pr_auc:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average Precision-Recall Curve for all classes')
    plt.legend(loc="lower left")
    plt.savefig('./images/PRC.png')
    plt.show()
    print('PRC曲线已绘制完成')


# 绘制模型平均ROC曲线
plot_average_roc()


# 绘制模型平均PRC曲线
plot_average_prc()