import numpy as np
class Var():
    '''
    This class defines a multivariate dual number
    '''
    def __init__(self, val, dual_paras):
        """ constructor for Var class

        INPUT
        =======
        val: value of the input variable
        dual_paras: partial derivatives with respect to each input variable
        
        RETURNS
        =======
        Var object: self.val and self.dir

        EXAMPLES
        =======
        >>> a = Var(1, np.array([1]))
        >>> print(a.val, a.der)
        1 [1]
        """
        self.val = val
        self.der = dual_paras
    
    def __add__(self, other):
        """ returns a Var as the result of self + other

        INPUT
        =======
        self: a Var object (object before +)
        other: a Var object or a real number (object after +)
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(2, np.array([1]))
        >>> t = x + 3
        >>> print(t.val, t.der)
        5 [1]
        
        >>> x = Var(3, np.array([1,0]))
        >>> y = Var(2, np.array([0,1]))
        >>> z = Var(3, np.array([1,0]))
        >>> z1 = x + y
        >>> print('x + y: {}'.format(vars(z1)))
        x + y: {'val': 5, 'der': array([1, 1])}
        >>> z2 = x + 1
        >>> print('x + 1: {}'.format(vars(z2)))
        x + 1: {'val': 4, 'der': array([1, 0])}
        >>> z3 = 1 + x
        >>> print('1 + x: {}'.format(vars(z3)))
        1 + x: {'val': 4, 'der': array([1, 0])}
        
        """
        try: # two Var objects
            value = self.val + other.val
            der = self.der + other.der
            return Var(value, der)
        except AttributeError: # Var + real number
            return Var(self.val + other, self.der)
    
    def __radd__(self, other):
        """ return a Var as the result of other + self

        INPUT
        =======
        self: a Var object (object after +)
        other: a Var object or a real number (object before +)
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(2, np.array([1]))
        >>> t = 3 + x
        >>> print(t.val, t.der)
        5 [1]
        """
        return self + other

    def __mul__(self, other):
        """ returns a Var as the result of self * other

        INPUT
        =======
        self: a Var object (object before *)
        other: a Var object or a real number (object after *)

        RETURNS
        =======
        Var object: a new Var object with new val and ders

        EXAMPLES
        =======
        >>> x = Var(3, np.array([1,0]))
        >>> y = Var(2, np.array([0,1]))
        >>> z = Var(3, np.array([1,0]))
        >>> z4 = y*2
        >>> z5 = 2*y
        >>> z6 = -1*y
        >>> z7 = y*(-1)
        >>> z8 = x*y
        >>> print(vars(z4))
        {'val': 4, 'der': array([0, 2])}
        >>> print(vars(z5))
        {'val': 4, 'der': array([0, 2])}
        >>> print(vars(z6))
        {'val': -2, 'der': array([ 0, -1])}
        >>> print(vars(z7)) 
        {'val': -2, 'der': array([ 0, -1])}
        >>> print(vars(z8))  
        {'val': 6, 'der': array([2, 3])}
        """
        try: # two Var objects
            value = self.val * other.val
            der = self.val*other.der + other.val * self.der
            return Var(value, der)
        except AttributeError: # Var * real number
            return Var(self.val*other, self.der * other)

    def __rmul__(self, other):
        """ returns a Var as the result of other * self
        
        INPUT
        =======
        self: a Var object (object after *)
        other: a Var object or a real number (object before *)
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(2, np.array([1]))
        >>> t = 3 * x
        >>> print(t.val, t.der)
        6 [3]
        """
        return self * other
    
    def __sub__(self, other):
        """ returns a Var as the result of self - other
    
        INPUT
        =======
        self: a Var object (object before -)
        other: a Var object or a real number (object after -)
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(3, np.array([1,0]))
        >>> y = Var(2, np.array([0,1]))
        >>> z = Var(3, np.array([1,0]))
        >>> z1 = x - y
        >>> print('x - y: {}'.format(vars(z1)))
        x - y: {'val': 1, 'der': array([ 1, -1])}
        >>> z2 = x - 2
        >>> print('x - 2: {}'.format(vars(z2)))
        x - 2: {'val': 1, 'der': array([1, 0])}
        >>> z3 = 2 - x
        >>> print('2 - x: {}'.format(vars(z3)))
        2 - x: {'val': -1, 'der': array([-1,  0])}
        """
        try: # two Var objects
            value = self.val - other.val
            der = self.der - other.der
            return Var(value, der)
        except AttributeError: # Var - real number
            return Var(self.val-other, self.der)

    def __rsub__(self, other):
        """ returns a Var as the result of other - self
        
        INPUT
        =======
        self: a Var object (object after -)
        other: a Var object or a real number (object before -)
            
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(2, np.array([1]))
        >>> t = 3 - x
        >>> print(t.val, t.der)
        1 [-1]
        """
        return -1 *(self-other)

    def __pow__(self, other):
        """ returns a Var as the result of self**(other)
        
        INPUT
        =======
        self: a Var object (object before **)
        other: a Var object or a real number (object after **)
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(3, np.array([1,0]))
        >>> y = Var(2, np.array([0,1]))
        >>> z = Var(3, np.array([1,0]))
        >>> z1 = x**y
        >>> print('x ** y: {}'.format(vars(z1)))
        x ** y: {'val': 9, 'der': array([6.       , 9.8875106])}
        >>> z2 = x**2
        >>> print('x ** 2: {}'.format(vars(z2)))
        x ** 2: {'val': 9, 'der': array([6, 0])}
        """
        
        try: # two Var objects 
        # d(a**c)/dx = d(a**c)/da * (da / dx) + d(a**c)/dc * (dc / dx) 
        # = c*(a**(c-1)) * (da / dx) + a**c*ln(a) * (dc / dx) 
            value = self.val**other.val
            der = other.val * (self.val ** (other.val - 1)) * self.der + value * np.log(self.val) * other.der
            return Var(value, der)
        except AttributeError: # Var ** real number
            return Var(self.val**other, other * (self.val ** (other-1)) * self.der)

    def __rpow__(self, other):
        """ returns a Var as the result of other**(self)

        INPUT
        =======
        self: a Var object (object after **)
        other: a Var object or a real number (object before **)
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(3, np.array([1,0]))
        >>> y = Var(2, np.array([0,1]))
        >>> z = Var(3, np.array([1,0]))
        >>> z1 = x**y
        >>> print('x ** y: {}'.format(vars(z1)))
        x ** y: {'val': 9, 'der': array([6.       , 9.8875106])}
        >>> z2 = x**2
        >>> print('x ** 2: {}'.format(vars(z2)))
        x ** 2: {'val': 9, 'der': array([6, 0])}
        >>> z3 = 2**x
        >>> print('2 ** x: {}'.format(vars(z3)))
        2 ** x: {'val': 8, 'der': array([5.54517744, 0.        ])}
        >>> z4 = x**(-1)
        >>> print('x ** (-1): {}'.format(vars(z4)))
        x ** (-1): {'val': 0.3333333333333333, 'der': array([-0.11111111, -0.        ])}
        """
        
        # the only scenario using this is when other is a real number and self is a Var object
        value = other **self.val
        # d(o ** s)/dx = o**s *log(o)*( ds/dx)
        der = value * np.log(other) * self.der
        return Var(value, der)

    def __truediv__(self, other):
        """ returns a Var as the result of self / other

        INPUT
        =======
        self: a Var object (numerator)
        other: a Var object or a real number (denominator)
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        ======= 
        >>> x = Var(3, np.array([1,0]))
        >>> y = Var(2, np.array([0,1]))
        >>> p = x * (y * (-1))
        >>> print(p.val, p.der)
        -6 [-2 -3]
        
        """
        return self * (other ** (-1))
    
    def __rtruediv__(self, other):
        """ returns a Var as the result of other / self

        INPUT
        =======
        self: a Var object (numerator)
        other: a Var object or a real number (denominator)
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(2, np.array([1,0]))
        >>> y = 2
        >>> v = y*(x**(-1))
        >>> print(v.val, v.der)
        1.0 [-0.5 -0. ]
        """
        return other*(self**(-1))
    
    def __neg__(self):
        """ returns a Var as the result of - self

        INPUT
        =======
        self: a Var object
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(2, np.array([1]))
        >>> p = (-1) * x
        >>> print(p.val, p.der)
        -2 [-1]
        """
        return (-1)*self

    def __pos__(self):
        """ returns a Var as the result of + self

        INPUT
        =======
        self: a Var object
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(2, np.array([1]))
        >>> print(x.val, x.der)
        2 [1]
        """
        return Var(self.val, self.der)

    def __eq__(self, other):
        """ returns the result of self == other

        INPUT
        =======
        self: a Var object (before ==)
        other: a Var object or something else(after ==)
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(3, np.array([1,0]))
        >>> y = Var(2, np.array([0,1]))
        >>> z = Var(3, np.array([1,0]))
        >>> print(x==y)
        False
        >>> print(x == z)
        True
        """
        try:
            return (self.val == other.val) & (list(self.der) == list(other.der))
        except AttributeError:
            return False

    def __ne__(self, other):
        """ returns a the result of self != other

        INPUT
        =======
        self: a Var object (before !=)
        other: a Var object or something else(after !=)
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(3, np.array([1,0]))
        >>> y = Var(2, np.array([0,1]))
        >>> z = Var(3, np.array([1,0]))
        >>> print(x!=y)
        True
        >>> print(x != z)
        False
        """
        try:
            return (self.val != other.val) | (list(self.der) != list(other.der))
        except AttributeError:
            return True

    @staticmethod
    def log(var):
        """ returns a Var as the result of var.log()

        INPUT
        =======
        var: a Var object or real number
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(3, np.array([1,0]))
        >>> z1 = Var.log(x)
        >>> print('log(x): {}'.format(vars(z1)))
        log(x): {'val': 1.0986122886681098, 'der': array([0.33333333, 0.        ])}
        """
        try:
            val = np.log(var.val)
            der = np.array(list(map(lambda x: x / var.val, var.der)))
            return Var(val, der)
        except AttributeError:
            return np.log(var)
        
    @staticmethod
    def logk(var, k):
        """ returns a Var as the result of var.logk()

        INPUT
        =======
        var: a Var object or a real number
        k: the base for log
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(3, np.array([1,0]))
        >>> z1 = Var.logk(x, 3.0)
        >>> print('logk(x, 3.0): {}'.format(vars(z1)))
        logk(x, 3.0): {'val': 1.0, 'der': array([0.30341308, 0.        ])}
        """
        try:
            val = np.log(var.val) / np.log(k)
            der = np.array(list(map(lambda x: x / var.val * (1/np.log(k)), var.der)))
            return Var(val, der)
        except AttributeError:
            return np.log(var) / np.log(k)
        
    @staticmethod
    def exp(var):
        """ returns a Var as the result of var.exp()

        INPUT
        =======
        var: a Var object or a real number
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(3, np.array([1,0]))
        >>> z1 = Var.exp(x)
        >>> print('exp(x): {}'.format(vars(z1)))
        exp(x): {'val': 20.085536923187668, 'der': array([20.08553692,  0.        ])}
        """
        try:
            val = np.exp(var.val)
            der = np.array(list(map(lambda x: x * val, var.der)))
            return Var(val, der)
        except AttributeError:
            return np.exp(var)

    @staticmethod
    def sqrt(var):
        """ returns a Var as the result of var.sqrt()

        INPUT
        =======
        var: a Var object or a real number
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(3, np.array([1,0])) 
        >>> z1 = Var.sqrt(x)
        >>> print('sqrt(x): {}'.format(vars(z1)))
        sqrt(x): {'val': 1.7320508075688772, 'der': array([0.28867513, 0.        ])}
        """
        try:
            val = np.sqrt(var.val)
            der = np.array(list(map(lambda x: 0.5 * (var.val ** (-0.5)) * x, var.der)))
            return Var(val, der)
        except AttributeError:
            return np.sqrt(var)

    @staticmethod
    def sin(var):
        """ returns a Var as the result of var.sin()

        INPUT
        =======
        var: a Var object or a real number
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(3, np.array([1,0])) 
        >>> z1 = Var.sin(x)
        >>> print('sin(x): {}'.format(vars(z1)))
        sin(x): {'val': 0.1411200080598672, 'der': array([-0.9899925, -0.       ])}
        """
        try:
            val = np.sin(var.val)
            der = np.array(list(map(lambda x: np.cos(var.val) * x, var.der)))
            return Var(val, der)
        except AttributeError:
            return np.sin(var)

    @staticmethod
    def cos(var):
        """ returns a Var as the result of var.cos()

        INPUT
        =======
        var: a Var object or a real number
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(3, np.array([1,0])) 
        >>> z1 = Var.cos(x)
        >>> print('cos(x): {}'.format(vars(z1)))
        cos(x): {'val': -0.9899924966004454, 'der': array([-0.14112001, -0.        ])}
        """
        try:
            val = np.cos(var.val)
            der = np.array(list(map(lambda x: -np.sin(var.val) * x, var.der)))
            return Var(val, der)
        except AttributeError:
            return np.cos(var)
        

    @staticmethod
    def tan(var):
        """ returns a Var as the result of var.tan()

        INPUT
        =======
        var: a Var object or a real number
        
        RETURNS
        =======
        Var object: a new Var object with new val and ders
        
        EXAMPLES
        =======
        >>> x = Var(3, np.array([1,0])) 
        >>> z1 = Var.tan(x)
        >>> print('tan(x): {}'.format(vars(z1)))
        tan(x): {'val': -0.1425465430742778, 'der': array([1.02031952, 0.        ])}
    
        """
        try:
            val = np.tan(var.val)
            der = np.array(list(map(lambda x: 1 / (np.cos(var.val) ** 2) * x, var.der)))
            return Var(val, der)
        except AttributeError:
            return np.tan(var)

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

    # x = Var(3, np.array([1,0]))
    # y = Var(2, np.array([0,1]))
    # z = Var(3, np.array([1,0]))

    # # eq, ne
    # print(x==y)
    # print(x == z)
    # print(x!=y)
    # print(x != z)

    # # neg
    # z1 = -x
    # print('-x: {}'.format(vars(z1)))
    # z2 = -(x**2)
    # print('-x**2: {}'.format(vars(z2)))

    # # div
    # z1 = x / y
    # print('x / y: {}'.format(vars(z1)))
    # z2 = x / 2
    # print('x / 2: {}'.format(vars(z2)))

    # # pow
    # z1 = x**y
    # print('x ** y: {}'.format(vars(z1)))
    # z2 = x**2
    # print('x ** 2: {}'.format(vars(z2)))
    # z3 = 2**x
    # print('2 ** x: {}'.format(vars(z3)))
    # z4 = x**(-1)
    # print('x ** (-1): {}'.format(vars(z4)))

    # # sub
    # z1 = x - y
    # print('x - y: {}'.format(vars(z1)))
    # z2 = x - 2
    # print('x - 2: {}'.format(vars(z2)))
    # z3 = 2 - x
    # print('2 - x: {}'.format(vars(z3)))


    # # add
    # z1 = x + y
    # print('x + y: {}'.format(vars(z1)))
    # z2 = x + 1
    # print('x + 1: {}'.format(vars(z2)))
    # z3 = 1 + x
    # print('1 + x: {}'.format(vars(z3)))

    # # mul
    # z4 = y*2
    # print('y * 2: {}'.format(vars(z4)))
    # z5 = 2*y
    # print('2 * y: {}'.format(vars(z5)))
    # z6 = -1*y
    # print('-1 * y: {}'.format(vars(z6)))
    # z7 = y*(-1)
    # print('y * (-1): {}'.format(vars(z7)))
    # z8 = x*y
    # print('x * y: {}'.format(vars(z8)))


    # x = Var(3, np.array([1]))

    # # log
    # z1 = Var.log(x)
    # print('log(x): {}'.format(vars(z1)))

    # # logk
    # z1 = Var.logk(x, 3.0)
    # print('logk(x, 3.0): {}'.format(vars(z1)))

    # # exp
    # z1 = Var.exp(x)
    # print('exp(x): {}'.format(vars(z1)))
    
    # # sqrt
    # z1 = Var.sqrt(x)
    # print('sqrt(x): {}'.format(vars(z1)))
    
    # # sin
    # z1 = Var.sin(x)
    # print('sin(x): {}'.format(vars(z1)))
    
    # # cos
    # z1 = Var.cos(x)
    # print('cos(x): {}'.format(vars(z1)))
    
    # # tan
    # z1 = Var.tan(x)
    # print('tan(x): {}'.format(vars(z1)))
    