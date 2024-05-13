# 一个标准的深度学习模型预测方法流程

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# CNN的输入是四个维度：第一个是样本的数量，通常shape[0]表示样本的数量，第二个维度是通道，如果是灰度图就是一个通道，第三、四是height 和 width 表示图像的高度和宽度
# (batch_size, channels, height, width)
X = X.reshape((X.shape[0],1,8,8))

# random_state是随机数，一组随机数抽两个数据，分别是从X和y中抽取，也就是图像对应着的target
# train_size优先级大于test_size,如果同时设置是按照train_size
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=42)

# 转换数据格式为numpy,float是转换成浮点型，long是转换成整数型
# numpy不用于图计算
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

# 创建DataLoader
# 相当于将图像和标签进行绑定操作，一一对应
train_dataset = TensorDataset(X_train,y_train)
test_dataset = TensorDataset(X_test,y_test)
# Dataloader里面有一个生成器, 每次记录运行到哪了，不会占用内存，batch_size是一次性处理图像的数量，shuffle表示是否随机打乱
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN,self).__init__()
        self.fc1 = nn.Linear(1*8*8,64)
        self.fc2 = nn.Linear(64,10)

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x = torch.relu()
