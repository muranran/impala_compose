import os
import re
import tensorflow.compat.v1 as tf

# test hasattr
class A:
    def __init__(self):
        self.ec = None
        pass

    @property
    def b(self):
        return self.ec


def test_hasattr():
    a = A()
    print(a.__dict__)
    print("a has attr {} : {}".format("ec", hasattr(a, "ec")))
    print("a has attr {} : {}".format("b", hasattr(a, "b")))

    # print("{} | {}".format(a))


if __name__ == '__main__':
    test_hasattr()
    pass
