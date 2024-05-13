# with 又称上下文管理器，在处理文件时，无论是否产生异常，都能保证with语句执行完毕后关闭已经打开的文件，这个过程是自动的，无需手动操作

def my_write(filename):
    with open(filename,'w',encoding='utf-8') as file:
        file.write('北京欢迎你')

def my_read(filename):
    with open(filename,'r',encoding='utf-8') as file:
        print(file.read())

def my_copy(src_file,target_file):
    with open(src_file,'r',encoding='utf-8') as file:
        with open(target_file,'w',encoding='utf-8') as file1:
            file1.write(file.read()) #将读取的文件直接写入


if __name__ == '__main__':
    my_write('data')
    my_read('data')
    my_copy('data','./copy_data')