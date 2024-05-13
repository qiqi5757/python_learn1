# 高级数据
# 使用key-value方式进行组织数据，内置的json模块专门用于处理JSON（JaveScript Object Notation）格式的数据
# 不需要进行类型转换就可以了，文件的基本操作
# json.dumps(obj):将Python数据类型转成JSON格式过程，编码过程
# json.loads(s):将JSON格式字符串转换成Python数据类型，解码过程
# json.dump(obj,file):与dumps()功能相同，将转换结果存储到文件file中
# json.load(file):与loads()功能相同，从文件file中都读入数据

import json
# 准备高维数据
lst = [
    {'name':'小红','age':12,'score':12},
    {'name':'小明','age':14,'score':42},
    {'name':'小兰','age':16,'score':35},
]

# ensure_ascii正常显示中文，indent美观用的，增加数据的缩进，json格式的字符串更具有可读性
s = json.dumps(lst,ensure_ascii=False,indent=4)

print(type(s))
print(s)

s2 = json.loads(s)
print(type(s2))
print(s2)

# 编码进文件中
with open('student.txt','w') as file:
    json.dump(lst,file,ensure_ascii=False,indent=4)

# 解码到程序中
with open('student.txt','r') as file:
    lst2 = json.load(file) # 直接是列表类型，不用进行转换
    print(type(lst2))
    print(lst2)