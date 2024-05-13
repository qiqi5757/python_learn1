class Father(object):
    def __init__(self,name):
        self.name = name

    def show(self):
        print(f'我的名字{self.name}')

class Mother(object):
    def __init__(self,age):
        self.age = age

    def show1(self):
        print(f'我的年龄{self.age}')

class Son(Father,Mother):
    def __init__(self,name,age,id):
        Father.__init__(self,name)
        Mother.__init__(self,age)
        self.id = id

    def show1(self):
        print(f'我的名字{self.name},我的年龄{self.age},我的id{self.id}')

son = Son('小明',18,1234)
#son.show()
son.show1()