# 封装
class young:

    # 单下划线
    def _fun1(self):
        print('单下划线')

    # 双下划线
    def __fun2(self):
        print('双下划线')

    # 首尾双下划线
    def __init__(self,name,age,id):
        self.name = name
        self._age = age
        self.__id = id

    # 实例方法
    def show(self):
        print(f'我的名字{self.name},我的年龄{self._age},我的学号{self.__id}')

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self,value):
        self.__id = value

people = young('张三',20,12345)
# print(people.name)
# print(people._age)
# #print(people.__id)
# people.show()
# people._fun1()
# people._young__fun2()
# #print(dir(young))
people.id = 2345
print(people.id)