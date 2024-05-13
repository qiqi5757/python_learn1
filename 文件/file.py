# 打开文件,如果文件不存在，则会创建一个文件
def my_write(s):
    # 这里的model可以是r和w
    file = open('new_file.txt','a',encoding='utf-8')

    # 写入文件要是字符串类型的
    file.write(s)
    file.close()

def my_write_list(file,list):
    f = open(file,'a',encoding='utf-8')
    # 追加文件
    f.writelines(list)
    f.close()

if __name__ == '__main__':
    my_write('写入文件')
    list = ['姓名\t','年龄\t','成绩\n','张三','30','23']
    my_write_list('new_file.txt',list)