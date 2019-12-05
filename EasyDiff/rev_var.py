import numpy as np
import pytest
class Var():
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
        try: # two Var objects
            z = Var(self.value + other.value)
            self.children.append((1.0, z)) # weight = ∂z/∂self = 1
            other.children.append((1.0, z)) # weight = ∂z/∂other = 1
            return z
        except AttributeError: # Var + real number
            z = Var(self.value + other)
            self.children.append((1.0, z))
            return z

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        try: # two Var objects
            z = Var(self.value * other.value)
            self.children.append((other.value, z)) # weight = ∂z/∂self = other.value
            other.children.append((self.value, z)) # weight = ∂z/∂other = self.value
            return z
        except AttributeError: # Var + real number
            z = Var(self.value * other)
            self.children.append((other, z))
            return z

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        pass
    
    def __rsub__(self, other):
        pass
    
    def __pow__(self, other):
        pass
    
    def __rpow__(self, other):
        pass

    def __truediv__(self, other):
        pass
    
    def __rtruediv__(self, other):
        pass
    
    def __neg__(self):
        pass
    
    def __pos__(self):
        pass

    def __eq__(self, other):
        pass

    def __ne__(self, other):
        pass

    @staticmethod
    def log(var):
        pass
        
    @staticmethod
    def logk(var, k):
        pass
    
    @staticmethod
    def exp(var):
        pass
    
    @staticmethod
    def sqrt(var):
        pass
    
    @staticmethod
    def sin(var):
        try:
            z = Var(np.sin(var.value))
            var.children.append((np.cos(var.value), z)) # weight = ∂z/∂var = cos(var.value)
            return z
        except:
            return np.sin(var)

    @staticmethod
    def cos(var):
        try:
            z = Var(np.cos(var.value))
            var.children.append((-np.sin(var.value), z)) # weight = ∂z/∂var = -sin(var.value)
            return z
        except:
            return np.cos(var)
    
    @staticmethod
    def tan(var):
        try:
            z = Var(np.tan(var.value))
            var.children.append((1 / (np.cos(var.value) ** 2), z)) # weight = ∂z/∂var = 1/(np.cos(var.value)^2)
            return z
        except:
            return np.tan(var)
    

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

    x = Var(0.5)
    y = Var(4.2)
    z = x * y + Var.sin(x)
    z.grad_value = 1.0

    assert z.value == pytest.approx(2.579425538604203)
    assert x.grad() == pytest.approx(y.value + np.cos(x.value))
    assert y.grad() == pytest.approx(x.value)