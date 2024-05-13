# 代码主要是用于构建简单的全连接类对手写数字数据进行分类
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

# transform=transform 表示将一个已经定义好的数据转换管道 transform 赋值给变量 transform。
# 在这段代码中，transform 是一个数据转换管道，它包含了一系列的数据转换操作，如将图像转换为张量、对张量进行归一化等。
# 通过将 transform 赋值给变量 transform，我们可以在定义数据集时直接使用这个数据转换管道，从而应用这些转换操作到数据集中的图像。这种写法可以使代码更加简洁和清晰。
train_dataset = torchvision.datasets.MNIST(root='data',train=True,transform = transforms,download=True)
test_dataset = torchvision.datasets.MNIST(root='data',train=False,transform= transforms,download=True)

train_loader = torch.utils.data.DataLoaders(dataset = train_dataset,batch_size = 100,shuffle = True)
test_loader = torch.utils.data.DataLoaders(dataset = test_dataset,batch_size = 100,shuffle = False)

class SimpleFC(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        # Linear里面有w和b的参数，可以调整
        self.fc1 = nn.Linear(input_size,hidden_size) # y = wx+b
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        # 2*4*3 2*12
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 每个图片的大小事28*28，隐藏层为通常为2的幂次方，128，256等，输出层就是10，因为是10个数字的类别
model = SimpleFC(28*28,128,10)
# 计算损失，通常将模型的输出与真实标签传递给损失函数，然后通过反向传播来更新模型参数，以最小化损失函数，从而实现模型的训练。
criterion = nn.CrossEntropyLoss()
# 优化器（optimizer）用于根据模型的损失函数计算的梯度来更新模型的参数，以最小化损失函数并提高模型的性能，Adam 是一种常用的优化算法，它是一种自适应学习率优化算法，通常在训练神经网络时表现良好
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

total_step = len(train_loader)
num_epochs = 5
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        #
        outputs = model(images)
        loss = criterion(outputs,labels)
        # 计算梯度之前，需要将之前计算的梯度清零，以避免梯度累积的影响
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()
        # 及时监控模型的训练情况
        if (i+1)%100 == 0:
            print('Epoch[{}/{}],Step[{}/{}],Loss:{:.4f'.format(epoch+1,num_epochs,i+1,total_step,loss.item()))

# 测试模型
model.eval()
# 在评估模式下，模型会禁用这些功能，并进行一些其他的调整，以便在验证集或测试集上进行模型的评估
with torch.no_grad:
    # 记录模型在测试集上正确预测的样本数量
    correct = 0
    # 记录测试集中总的样本数量
    total = 0
    for images,labels in test_loader:
        outputs = model(images)
        # 由于我们通常只关心最大值所在的索引，因此使用下划线 _ 来接收 max_values，表示我们不会使用这个值。而 predicted 则用来接收最大值所在的索引，即模型预测的类别。
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('测试集准确率：{}%'.format(100*correct/total))