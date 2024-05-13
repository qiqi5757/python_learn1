def my_write(filename):
    # 打开
    file= open(filename,'w+',encoding='utf-8')
    # 操作
    file.write('你好')
    # 修改指针的位置
    file.seek(0)
    # 读取
    # s = file.read() # 读取全部
    s = file.read(2) # 读取两个字符
    s = file.readline(1) # 读取一行
    s = file.readline(2) # 读取一行中前两个字符
    s = file.readline() #读取所有，一行为列表中的一个元素，s是列表类型
    # 如果想跳读，就移动指针的位置
    file.seek(3)
    s = file.read() # 读取全部
    print(type(s),s)

    file.close()

if __name__ == '__main__':
    my_write('new_file.txt')
