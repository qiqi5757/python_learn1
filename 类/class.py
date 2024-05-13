class young:
    # 类属性
    date = 54

    # 初始方法
    def __init__(self,xm,age):
        # 实例属性
        self.name = xm
        self.age = age

    # 实例方法
    def show(self):
        print(f'我的名字{self.name},我的年龄{self.age}')

    # 静态方法
    @staticmethod
    def sm():
        # print(self.name)
        # self.show
        print('我是静态方法，不能调用实例属性，不能调用实例方法')

    # 类方法
    @classmethod
    def cm(cls):
        # print(self.name)
        # self.show
        print('我是类方法，不能调用实例属性，不能调用实例方法')

people = young('小明',18)
#print(people.name,people.age)
# people.show()
# young.sm()
# young.cm()
#
# young.date = 56
# print(young.date)

people1 = young('小红',17)
#print(people1.name,people1.age)
people2 = young('张三',20)
#print(people2.name,people2.age)

lst = [people,people1,people2]
for i in lst:
    print(i.name,i.age)

# 动态绑定一个实例属性
people1.id = 1234
print(people1.name,people1.age,people1.id)

def method():
    print('我是一个普通的函数')
# 函数的赋值
people1.ff = method
# 调用
people1.ff()