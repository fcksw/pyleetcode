from typing import Protocol

class MyInterface(Protocol):  # 定义协议
    def my_method(self) -> None:
        ...

class MyClass:  # 无需继承，只需实现方法
    def my_method(self) -> None:
        print("Implementation of my_method")

    def test_method(self):
        print("aa")

def process(obj: MyInterface) -> None:  # 类型提示为协议
    obj.my_method()

process(MyClass())  # 类型检查通过