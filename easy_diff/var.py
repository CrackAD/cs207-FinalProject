import numpy as np

class Var():
    '''
    This class defines a multivariate dual number
    '''
    def __init__(self, val, dual_paras):
        ''' constructor for Var class

        args:
            val: value of the input variable
            dual_paras: partial derivatives with respect to each input variable
        '''
        self.val = val
        self.der = dual_paras
    
    def __add__(self, other):
        ''' returns a Var as the result of self + other

        args:
            self: a Var object (object before +)
            other: a Var object or a real number (object after +)
        '''
        try: # two Var objects
            value = self.val + other.val
            der = self.der + other.der
            return Var(value, der)
        except AttributeError: # Var + real number
            return Var(self.val + other, self.der)
    
    def __radd__(self, other):
        ''' return a Var as the result of other + self

        args:
            self: a Var object (object after +)
            other: a Var object or a real number (object before +)
        '''
        return self + other

    def __mul__(self, other):
        ''' returns a Var as the result of self * other

        args:
            self: a Var object (object before *)
            other: a Var object or a real number (object after *)
        '''
        try: # two Var objects
            value = self.val * other.val
            der = self.val*other.der + other.val * self.der
            return Var(value, der)
        except AttributeError: # Var * real number
            return Var(self.val*other, self.der * other)

    def __rmul__(self, other):
        ''' returns a Var as the result of other * self

        args:
            self: a Var object (object after *)
            other: a Var object or a real number (object before *)
        '''
        return self * other
    
    def __sub__(self, other):
        ''' returns a Var as the result of self - other

        args:
            self: a Var object (object before -)
            other: a Var object or a real number (object after -)
        '''
        try: # two Var objects
            value = self.val - other.val
            der = self.der - other.der
            return Var(value, der)
        except AttributeError: # Var - real number
            return Var(self.val-other, self.der)

    def __rsub__(self, other):
        ''' returns a Var as the result of other - self

        args:
            self: a Var object (object after -)
            other: a Var object or a real number (object before -)
        '''
        return -1 *(self-other)

    def __pow__(self, other):
        ''' returns a Var as the result of self**(other)

        args:
            self: a Var object (object before **)
            other: a Var object or a real number (object after **)
        '''
        try: # two Var objects 
        # d(a**c)/dx = d(a**c)/da * (da / dx) + d(a**c)/dc * (dc / dx) 
        # = c*(a**(c-1)) * (da / dx) + a**c*ln(a) * (dc / dx) 
            value = self.val**other.val
            der = other.val * (self.val ** (other.val - 1)) * self.der + value * np.log(self.val) * other.der
            return Var(value, der)
        except AttributeError: # Var - real number
            return Var(self.val**other, other * (self.val ** (other-1)) * self.der)

    def __rpow__(self, other):
        ''' returns a Var as the result of other**(self)

        args:
            self: a Var object (object after **)
            other: a Var object or a real number (object before **)
        '''
        # the only scenario using this is when other is a real number and self is a Var object
        value = other **self.val
        # d(o ** s)/dx = o**s *log(o)*( ds/dx)
        der = value * np.log(other) * self.der
        return Var(value, der)

    def __div__(self, other):
        ''' returns a Var as the result of self / other

        args:
            self: a Var object (numerator)
            other: a Var object or a real number (denominator)
        '''
        return self * (other ** (-1))
    
    def __rdiv__(self, other):
        ''' returns a Var as the result of other / self

        args:
            self: a Var object (denominator)
            other: a Var object or a real number (numerator)
        '''
        return other*(self**(-1))
    
    def __neg__(self):
        ''' returns a Var as the result of - self

        args:
            self: a Var object
        '''
        return (-1)*self

    def __pos__(self):
        ''' returns a Var as the result of + self

        args:
            self: a Var object
        '''
        return Var(self.val, self.der)

    def __eq__(self, other):
        ''' returns a the result of self == other

        args:
            self: a Var object (before ==)
            other: a Var object or something else(after ==)
        '''
        try:
            return (self.val == other.val) & (list(self.der) == list(other.der))
        except AttributeError:
            return False

    def __ne__(self, other):
        ''' returns a the result of self != other

        args:
            self: a Var object (before !=)
            other: a Var object or something else(after !=)
        '''
        try:
            return (self.val != other.val) | (list(self.der) != list(other.der))
        except AttributeError:
            return True



if __name__ == "__main__":
    x = Var(3, np.array([1,0]))
    y = Var(2, np.array([0,1]))
    z = Var(3, np.array([1,0]))

    # eq, ne
    print(x==y)
    print(x == z)
    print(x!=y)
    print(x != z)

    # neg
    z1 = -x
    print('-x: {}'.format(vars(z1)))
    z2 = -(x**2)
    print('-x**2: {}'.format(vars(z2)))

    # div
    z1 = x / y
    print('x / y: {}'.format(vars(z1)))
    z2 = x / 2
    print('x / 2: {}'.format(vars(z2)))

    # pow
    z1 = x**y
    print('x ** y: {}'.format(vars(z1)))
    z2 = x**2
    print('x ** 2: {}'.format(vars(z2)))
    z3 = 2**x
    print('2 ** x: {}'.format(vars(z3)))
    z4 = x**(-1)
    print('x ** (-1): {}'.format(vars(z4)))

    # sub
    z1 = x - y
    print('x - y: {}'.format(vars(z1)))
    z2 = x - 2
    print('x - 2: {}'.format(vars(z2)))
    z3 = 2 - x
    print('2 - x: {}'.format(vars(z3)))


    # add
    z1 = x + y
    print('x + y: {}'.format(vars(z1)))
    z2 = x + 1
    print('x + 1: {}'.format(vars(z2)))
    z3 = 1 + x
    print('1 + x: {}'.format(vars(z3)))

    # mul
    z4 = y*2
    print('y * 2: {}'.format(vars(z4)))
    z5 = 2*y
    print('2 * y: {}'.format(vars(z5)))
    z6 = -1*y
    print('-1 * y: {}'.format(vars(z6)))
    z7 = y*(-1)
    print('y * (-1): {}'.format(vars(z7)))
    z8 = x*y
    print('x * y: {}'.format(vars(z8)))

        