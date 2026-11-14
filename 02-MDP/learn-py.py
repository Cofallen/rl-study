from functools import cached_property
from huggingface_hub import cached_assets_path
import numpy as np


def deco(func):
    def wrapper(*args, **kwargs):
        print("before")
        result = func(*args, **kwargs)
        print("after")
        return result
    return wrapper  # 可以改变函数作用位置

def llm(func):
    print("1")
    return func  # 装饰器必须返回，常用，不改变原函数本身，只是增加功能到前面

# 相当于 hello = llm(hello) ， 因此装饰器必须返回函数
@deco
def hello(name):
    print(f"hello {name}") 

hello("Tom")

# 属性系
class C:
    def __init__(self):
        self._x = 0

    @property                     # 把方法变成只读属性，函数-》只读参数，这个参数调用就变化是最新的
    def x(self): return self._x
    
    def y(self): return self._x

    @x.setter
    def change_x(self, v): self._x = v

    @x.deleter
    def delete_x(self): del self._x

    @cached_property
    def heavy(self): return sum(range(10**6))

c = C()
print(c._x)
print(c.x)
print(c.y())

c._x = 1
c.change_x(2)
print(c._x)
print(c.x)
print(c.y())


# class C:
#     @staticmethod                  # 不传 self/cls
#     def f(): return 1

#     @classmethod                   # 传 cls，常用于工厂
#     def create(cls): return cls()

#     @abstractmethod                # 配合 ABC，强制子类实现
#     def run(self): ...

#     @x.setter                      # 同上、


# 框架装饰器示例（需要时取消注释，并先安装/导入对应库）
# from flask import Flask
# app = Flask(__name__)
# @app.route("/")                  # Flask
# import pytest
# @pytest.fixture                  # pytest
# import torch
# @torch.no_grad()                 # 关梯度