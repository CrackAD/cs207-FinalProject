import math
import pytest

class Var:
    def __init__(self, value):
        self.value = value
        self.children = [] # store the <weight, Var> tuple for all its child. 
        self.grad_value = None

    def grad(self):
        if self.grad_value is None:
            self.grad_value = sum(weight * var.grad()
                                  for weight, var in self.children)
        return self.grad_value

    def __add__(self, other):
        z = Var(self.value + other.value)
        self.children.append((1.0, z))
        other.children.append((1.0, z))
        return z

    def __mul__(self, other):
        z = Var(self.value * other.value)
        self.children.append((other.value, z)) # weight = ∂z/∂self = other.value
        other.children.append((self.value, z)) # weight = ∂z/∂other = self.value
        return z

def sin(x):
    z = Var(math.sin(x.value))
    x.children.append((math.cos(x.value), z))
    return z

x = Var(0.5)
y = Var(4.2)
z = x * y + sin(x)
z.grad_value = 1.0

assert z.value == pytest.approx(2.579425538604203)
assert x.grad() == pytest.approx(y.value + math.cos(x.value))
assert y.grad() == pytest.approx(x.value)