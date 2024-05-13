import jieba
# 读取文件

with open('huawei.txt','r',encoding='utf-8') as file:
    s = file.read()
print(s)
# 分词，词出现的频率
lst = jieba.lcut(s)
print(lst)

# 去重操作
set1 = set(lst)   #使用集合去重

#
d = {} # key:词，value：出现的次数
for item in set1:
    if len(item)>=2:
        d[item]=0

for item in lst:
    if item in d:
        d[item] = d.get(item)+1


new_lst = []
for item in d:
    new_lst.append([item,d[item]])

# 列表排序
new_lst.sort(key=lambda x:x[1],reverse=True)
print(new_lst[0:11]) # 显示前十项

