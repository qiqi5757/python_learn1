# 用torch实现一个简单全连接类，并可以前向传播
import torch
import torch.nn as nn

class SimpleFC(nn.Module):
    def __init__(self,input_size,output_size,hidden_size):
        super(SimpleFC,self).__init__()
        # input_size是输入，hidden_size是输出，所以下面的forward方法只用传入一个值
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = 10
hidden_size = 10

model = SimpleFC(input_size,2,hidden_size)

input_data = torch.randn(32,input_size)

output_data = model.forward(input_data)

print(f'输出形状：{output_data.shape}')