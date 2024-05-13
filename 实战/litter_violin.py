# 这是10个类的预测值的小提琴图
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
            probabilities = nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probabilities.numpy())
            all_labels.extend(labels.numpy())

    # 将标签二值化
    all_labels = label_binarize(all_labels, classes=range(num_classes))
    all_probs = np.array(all_probs)

    # 计算平均ROC AUC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
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

# 假设 collect_model_probabilities 返回的是合适的一维概率数组
def collect_model_probabilities(model, test_loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probabilities.numpy().flatten())  # 确保数据是一维的
    return all_probs

# 确保每个模型的数据是一维的
fcnn_probs = collect_model_probabilities(fcnn_model, test_loader)
cnn_probs = collect_model_probabilities(cnn_model, test_loader)
lstm_probs = collect_model_probabilities(lstm_model, test_loader)

data = [fcnn_probs, cnn_probs, lstm_probs]  # 这里每个元素都应该是一维数组


# 准备绘制小提琴图
plt.figure(figsize=(10, 6))
model_names = ['FCNN', 'CNN', 'LSTM']
data = [fcnn_probs, cnn_probs, lstm_probs]

# 创建小提琴图
parts = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=True)

# 自定义颜色和样式
colors = ['blue', 'green', 'red']
for partname, part, color in zip(['bodies', 'cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars'], parts.values(), colors):
    if partname == 'bodies':
        for pc in part:
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
    else:
        part.set_edgecolor(color)

plt.xticks(ticks=[1, 2, 3], labels=model_names)
plt.ylabel('Probability')
plt.title('Model Prediction Probability Distributions Across All Classes')

plt.tight_layout()

# 保存小提琴图
plt.savefig('./images/violin_plot.png', bbox_inches='tight')
plt.show()
print('图片保存成功')