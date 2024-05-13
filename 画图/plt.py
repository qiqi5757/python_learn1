# 目的是画一个饼状图
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
df = pd.read_excel('data.xlsx')

# 设置横纵坐标
labels = df['name']
y = df['beijing']

plt.pie(x = y,labels=labels,autopct= '%1.1f%%',startangle=0)
plt.axis('equal')
plt.title('2024year')
plt.show()