# 一维数组 和 二维数组
def my_write():
    # 一维数据，可以使用列表和元组，集合，但是不能使用字典
    lst = ['张三','李四','王五']
    with open('data.txt','w',encoding='utf-8') as file:
        file.write(','.join(lst)) # 将列表转成字符串

def my_read():
    with open('data.txt','r') as file:
        s = file.read()
        lst = s.split(',')
        print(lst)

def my_write_table():
    lst1 = [
        [1,2,3],
        [1,3,4]
    ]
    with open('data.txt','w',encoding='utf-8') as file:
        for item in lst1:
            line = ','.join(item) # 将列表中元素用逗号进行拼接
            file.write(line)
            file.write('\n')

def my_read_table():
    data = [] # 存储读取的数据
    with open('data','r',encoding='utf-8') as file:
        lst = file.readline() #每一行是列表中的一个元素
        for item in lst:
            new_lst = item[:len(item)-1].split(',')
            data.append(new_lst)

if __name__ == '__main__':
    my_write()
    my_write_table()
    my_read_table()