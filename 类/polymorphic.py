class People1:
    def do(self):
        print('吃饭')

class People2:
    def do(self):
        print('喝水')

class People3:
    def do(self):
        print('打豆豆')

def fun(obj):
    obj.do()

a = People1()
b = People2()
c = People3()

fun(a)
fun(b)
fun(c)