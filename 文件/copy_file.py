# 文件的复制
def copy(src,new_path):
    # 文件的复制是边读边写的操作
    # 打开源文件
    file1 = open(src,'rb')
    # 打开目标文件
    file2 = open(new_path,'wb')
    # 开始复制
    s = file1.read() # 读文件1的
    file2.write(s) # 写入文件2

    # 关闭文件,先开的后关，后开的先关
    file2.close()
    file1.close()

if __name__ == '__main__':
    src = './new_file.txt'
    new_path = './data'
    copy(src,new_path)
    print('文件复制完毕')
