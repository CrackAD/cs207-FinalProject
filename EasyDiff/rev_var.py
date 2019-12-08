import numpy as np
import pytest
class Rev_Var():
    '''
    This class defines a multivariate reverse mode node
    '''
    def __init__(self, value):
        """ constructor for Rev_Var class

        INPUT
        =======
        value: value of the current variable
        
        RETURNS
        =======

        Rev_Var object: self.value, self.children [(weight, Rev_Var)], and self.grad_value
        
        EXAMPLES
        =======
        >>> a = Rev_Var(1)
        >>> print(a.value, a.children, a.grad_value)
        1 [] None
        
        """
        self.value = value
        self.children = [] # store the <weight, Rev_Var> tuple for all its child. 
        self.grad_value = None # store derivatives of final expression with respect to current variable

    def grad(self):
        """ calculate partial derivatives of the final expression with respect to current variable

        INPUT
        =======
        self: a Rev_Var object 
        
        RETURNS
        =======

        Rev_Var object: a Rev_Var object with updated partial derivative df/dself
        
        EXAMPLES
        =======
        >>> y = Rev_Var(3.0)
        >>> z = Rev_Var.expk(4.0, y)
        >>> z.grad_value = 1
        >>> print(y.grad(), 4**3*np.log(4))
        108.0 88.722839111673

        
        """
        if self.grad_value is None:
            self.grad_value = sum(weight * var.grad()
                                  for weight, var in self.children)
        return self.grad_value

    def __add__(self, other):
        """ returns a Rev_Var as the result of self + other

        INPUT
        =======
        self: a Rev_Var object (object before +)
        other: a Rev_Var object or a real number (object after +)
        
        RETURNS
        =======

        Rev_Var object: a new Rev_Var object with new value and no children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(2.0)
        >>> y = Rev_Var(3.0)    
        >>> z1 = x + y
        >>> print('x + y: {}'.format(vars(z1)))        
        x + y: {'value': 5.0, 'children': [], 'grad_value': None}
        >>> z2 = x + 1
        >>> print('x + 1: {}'.format(vars(z2)))
        x + 1: {'value': 3.0, 'children': [], 'grad_value': None}
        >>> z3 = 1 + x
        >>> print('1 + x: {}'.format(vars(z3)))
        1 + x: {'value': 3.0, 'children': [], 'grad_value': None}
        

        """
        try: # two Var objects
            z = Rev_Var(self.value + other.value)
            self.children.append((1.0, z)) # weight = dz/dself = 1
            other.children.append((1.0, z)) # weight = dz/dother = 1
            return z
        except AttributeError: # Var + real number
            z = Rev_Var(self.value + other)
            self.children.append((1.0, z))
            return z

    def __radd__(self, other):
        """ return a Rev_Var as the result of other + self

        INPUT
        =======
        self: a Rev_Var object (object after +)
        other: a Rev_Var object or a real number (object before +)
        
        RETURNS
        =======

        Rev_Var object: a new Rev_Var object with new value and no children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(2.0)
        >>> t = 3 + x
        >>> print(t.value, t.grad_value)
        5.0 None
        
        """
        return self + other

    def __mul__(self, other):
        """ returns a Rev_Var as the result of self * other

        INPUT
        =======
        self: a Rev_Var object (object before *)
        other: a Rev_Var object or a real number (object after *)

        RETURNS
        =======

        Rev_Var object: a new Rev_Var object with value and no children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(3.0)
        >>> y = Rev_Var(2.0)
        >>> z = Rev_Var(3.0)
        >>> z4 = y*2
        >>> z5 = 2*y
        >>> z6 = -1*y
        >>> z7 = y*(-1)
        >>> z8 = x*y
        >>> print(vars(z4))
        {'value': 4.0, 'children': [], 'grad_value': None}
        >>> print(vars(z5))
        {'value': 4.0, 'children': [], 'grad_value': None}
        >>> print(vars(z6))
        {'value': -2.0, 'children': [], 'grad_value': None}
        >>> print(vars(z7)) 
        {'value': -2.0, 'children': [], 'grad_value': None}
        >>> print(vars(z8))  
        {'value': 6.0, 'children': [], 'grad_value': None}

        """
        try: # two Var objects
            z = Rev_Var(self.value * other.value)
            self.children.append((other.value, z)) # weight = dz/dself = other.value
            other.children.append((self.value, z)) # weight = dz/dother = self.value
            return z
        except AttributeError: # Var * real number
            z = Rev_Var(self.value * other)
            self.children.append((other, z))
            return z

    def __rmul__(self, other):
        """ returns a Rev_Var as the result of other * self
        
        INPUT
        =======
        self: a Rev_Var object (object after *)
        other: a Rev_Var object or a real number (object before *)
        
        RETURNS
        =======

        Rev_Var object: a new Rev_Var object with new value and no children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(2.0)
        >>> t = 3 * x
        >>> print(t.value, t.grad_value)
        6.0 None
        
        """
        return self * other

    def __sub__(self, other):
        """ returns a Rev_Var as the result of self - other
    
        INPUT
        =======
        self: a Rev_Var object (object before -)
        other: a Rev_Var object or a real number (object after -)
        
        RETURNS
        =======

        Rev_Var object: a new Rev_Var object with new value and children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(3.0)
        >>> y = Rev_Var(2.0)
        >>> z = Rev_Var(3.0)
        >>> z1 = x - y
        >>> print('x - y: {}'.format(vars(z1)))
        x - y: {'value': 1.0, 'children': [], 'grad_value': None}
        >>> z2 = x - 2
        >>> print('x - 2: {}'.format(vars(z2)))
        x - 2: {'value': 1.0, 'children': [], 'grad_value': None}
        >>> z3 = 2 - x
        >>> print('2 - x: {}'.format(vars(z3)))
        2 - x: {'value': -1.0, 'children': [], 'grad_value': None}

        """
        
        try: # two Var objects
            z = Rev_Var(self.value - other.value)
            self.children.append((1.0, z)) # weight = dz/dself = 1
            other.children.append((-1.0, z)) # weight = dz/dother = -1
            return z
        except AttributeError: # Var - real number
            z = Rev_Var(self.value - other)
            self.children.append((1.0, z))
            return z

    def __rsub__(self, other):
        """ returns a Rev_Var as the result of other - self
        
        INPUT
        =======
        self: a Rev_Var object (object after -)
        other: a Rev_Var object or a real number (object before -)
            
        RETURNS
        =======

        Rev_Var object: a new Rev_Var object with new value and children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(2.0)
        >>> t = 3 - x
        >>> print(t.value, t.grad_value)
        1.0 None
        
        """
        return -1 *(self - other)

    def __pow__(self, other):
        """" returns a Rev_Var as the result of self**(other)
        
        INPUT
        =======
        self: a Rev_Var object (object before **)
        other: a Rev_Var object or a real number (object after **)
        
        RETURNS
        =======

        Rev_Var object: a new Rev_Var object with new value and children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(3.0)
        >>> y = Rev_Var(2.0)
        >>> z = Rev_Var(3.0)
        >>> z1 = x**y
        >>> print('x ** y: {}'.format(vars(z1)))
        x ** y: {'value': 9.0, 'children': [], 'grad_value': None}
        >>> z2 = x**2
        >>> print('x ** 2: {}'.format(vars(z2)))
        x ** 2: {'value': 9.0, 'children': [], 'grad_value': None}
        

        """
        try: # two Rev_Var objects
            val = self.value**other.value
            z = Rev_Var(val)
            self.children.append((other.value*(self.value **(other.value - 1)), z)) # weight = dz/dself
            other.children.append((val*np.log(self.value), z))
            return z
        except AttributeError: # Var ** real number
            z = Rev_Var(self.value ** other)
            self.children.append((other*(self.value**(other-1)), z))
            return z
   
    def __rpow__(self, other):
        """ returns a Rev_Var as the result of other**(self)

        INPUT
        =======
        self: a Rev_Var object (object after **)
        other: a Rev_Var object or a real number (object before **)
        
        RETURNS
        =======

        Rev_Var object: a new Rev_Var object with new value and children
        
        EXAMPLES
        =======
        >>> y = Rev_Var(3.0)
        >>> z3 = 2**y
        >>> z3.grad_value = 1
        >>> print(y.grad(), 2**3*np.log(2))
        5.545177444479562 5.545177444479562
        
        >>> x = Rev_Var(3.0)
        >>> y = Rev_Var(2.0)
        >>> z = Rev_Var(3.0)
        >>> z1 = x**y
        >>> print('x ** y: {}'.format(vars(z1)))
        x ** y: {'value': 9.0, 'children': [], 'grad_value': None}
        >>> z2 = x**2
        >>> print('x ** 2: {}'.format(vars(z2)))
        x ** 2: {'value': 9.0, 'children': [], 'grad_value': None}
        >>> z3 = 2**x
        >>> print('2 ** x: {}'.format(vars(z3)))
        2 ** x: {'value': 8.0, 'children': [], 'grad_value': None}
        >>> z4 = x**(-1)
        >>> print('x ** (-1): {}'.format(vars(z4)))
        x ** (-1): {'value': 0.3333333333333333, 'children': [], 'grad_value': None}
        

        """
        # the only scenario using this is when other is a real number and self is a Var object
        z = Rev_Var(other **self.value)
        self.children.append(((other**self.value) * np.log(other), z))
        return z

    def __truediv__(self, other):
        """ returns a Rev_Var as the result of self / other

        INPUT
        =======
        self: a Rev_Var object (numerator)
        other: a Rev_Var object or a real number (denominator)
        
        RETURNS
        =======

        Rev_Var object: a new Rev_Var object with new value and children
        
        EXAMPLES
        =======
        >>> y = Rev_Var(3.0)
        >>> x = Rev_Var(2.0)
        >>> z4 = x/y
        >>> z4.grad_value = 1
        >>> print(x.grad(), 1/3)
        0.3333333333333333 0.3333333333333333
        >>> print(y.grad(),-2/(3**2))
        -0.2222222222222222 -0.2222222222222222
    
        """
        return self * (other ** (-1))

    def __rtruediv__(self, other):
        """ returns a Rev_Var as the result of other / self

        INPUT
        =======
        self: a Rev_Var object (numerator)
        other: a Rev_Var object or a real number (denominator)
        
        RETURNS
        =======

        Rev_Var object: a new Rev_Var object with new value and children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(3.0)
        >>> y = Rev_Var(2.0)
        >>> p = x * (y * (-1))
        >>> print(p.value, p.grad_value)
        -6.0 None
        
        """
        return other*(self**(-1))
    
    def __neg__(self):
        """ returns a Rev_Var as the result of - self

        INPUT
        =======
        self: a Rev_Var object
        
        RETURNS
        =======
        Rev_Var object: a new Rev_Var object with new value and children
        
        EXAMPLES
        =======
        >>> z6 = x/2
        >>> z7 = x/2
        >>> print(z6 != z7)
        False
        >>> x = Rev_Var(2.0)
        >>> x2 = Rev_Var(2.0)
        >>> print(x != x2)
        False
        >>> y = Rev_Var(3.0)
        >>> print(x != y)
        True
        
        """
        ans= Rev_Var(-self.value)
        ans.children = [(-1, self)]
        if self.grad_value:
            ans.grad_value = -self.grad_value
        return ans

    
    def __pos__(self):
        """ returns a Rev_Var as the result of + self

        INPUT
        =======
        self: a Rev_Var object
        
        RETURNS
        =======
        Rev_Var object: a Rev_Var object
        
        EXAMPLES
        =======
        >>> x = Rev_Var(3.0)
        >>> y = Rev_Var(2.0)
        >>> print(x.value, x.grad_value)
        3.0 None
        >>> print(y.value, y.grad_value)
        2.0 None

        """
        return self.copy()

    def __eq__(self, other):
        """ returns the result of self == other

        INPUT
        =======
        self: a Rev_Var object (before ==)
        other: a Rev_Var object or something else(after ==)
        
        RETURNS
        =======
        a boolean value
        
        EXAMPLES
        =======
        >>> z6 = x/2
        >>> z7 = x/2
        >>> print(z6 == z7)
        True
        >>> x2 = Rev_Var(2.0)
        >>> print(x == x2)
        False
        
        """
        try:
            # check equal value, derivative, and number of children
            condition1 = (self.value == other.value) and (self.grad_value == other.grad_value) and (len(self.children) == len(other.children))
            condition2 = True
            for c_self, c_other in zip(self.children, other.children):
                #check equal weights
                if c_self[0] != c_other[0]:
                    condition2 =  False
                    break
            return (condition1 and condition2)
        except AttributeError:
            return False

    def __ne__(self, other):
        """ returns a the result of self != other

        INPUT
        =======
        self: a Rev_Var object (before !=)
        other: a Rev_Var object or something else(after !=)
        
        RETURNS
        =======
        a boolean value
        
        EXAMPLES
        =======
        >>> x = Rev_Var(3.0)
        >>> y = Rev_Var(2.0)
        >>> z = Rev_Var(3.0)
        >>> print(x!=y)
        True
        >>> print(x != z)
        False
        """
        try:
            return not (self == other)
        except AttributeError:
            return True

    @staticmethod
    def log(var): # ln()
        """ returns a Rev_Var as the result of var.log()

        INPUT
        =======
        var: a Rev_Var object or real number
        
        RETURNS
        =======
        Rev_Var object: a new Rev_Var object with new val and children

        EXAMPLES
        =======        
        >>> x = Rev_Var(2.0)
        >>> z1 = Rev_Var.log(x)
        >>> z1.grad_value = 1
        >>> print(x.grad(), 1/2)
        0.5 0.5
        
        """
        try: # a Var object
            z = Rev_Var(np.log(var.value))
            var.children.append((1/ var.value, z))
            return z
        except AttributeError: # a real number
            return np.log(var)

    @staticmethod
    def logk(var, k): #logk(var)
        """ returns a Rev_Var as the result of var.log(k)

        INPUT
        =======
        var: a Rev_Var object or real number
        
        RETURNS
        =======
        Rev_Var object: a new Rev_Var object with new val and children
    
        EXAMPLES
        =======
        >>> x = Rev_Var(2.0)
        >>> z2 = Rev_Var.logk(x, 3.0)
        >>> z2.grad_value = 1
        >>> print(z2.value, np.log(2) / np.log(3))
        0.6309297535714574 0.6309297535714574
        >>> print(x.grad(), 1/(2*np.log(3)))
        0.45511961331341866 0.45511961331341866


        """
        try: # a Var object
            z = Rev_Var(np.log(var.value) / np.log(k))
            var.children.append((1/ (var.value*np.log(k)), z))
            return z
        except AttributeError: # a real number
            return np.log(var) / np.log(k)
    
    @staticmethod
    def exp(var): #e^(var)
        """ returns a Var as the result of Var.exp(var)

        INPUT
        =======
        var: a Rev_Var object or a real number
        
        RETURNS
        =======
        Rev_Var object: a new Rev_Var object with new value and children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(2.0)
        >>> z1 = Rev_Var.exp(x)
        >>> z1.grad_value = 1
        >>> print(x.grad(), np.exp(2))
        7.38905609893065 7.38905609893065
        >>> print(z1.value, np.exp(2))
        7.38905609893065 7.38905609893065

        """
        try:
            z = Rev_Var(np.exp(var.value))
            var.children.append((np.exp(var.value), z))
            return z
        except AttributeError: # two real numbers
            return np.exp(var)
    
    @staticmethod
    def expk(var,k): #k^(var)
        """ returns a Var as the result of k ** (var)

        INPUT
        =======
        var: a Var object or a real number
        k: the base of the exponential
        
        RETURNS
        =======
        Rev_Var object: a new Rev_Var object with new value and children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(3.0)
        >>> y = Rev_Var(2.0)
        >>> z = Rev_Var(3.0)
        >>> z1 = Rev_Var.exp(x)
        >>> z2 = Rev_Var.exp(y)
        >>> print('exp(x): {}'.format(vars(z1)))
        exp(x): {'value': 20.085536923187668, 'children': [], 'grad_value': None}
        >>> print('exp(y): {}'.format(vars(z2)))
        exp(y): {'value': 7.38905609893065, 'children': [], 'grad_value': None}
        
        >>> z = Rev_Var.expk(x, 2)
        >>> z.grad_value = 1
        >>> x.grad() == pytest.approx(2**3*np.log(2))
        True
        >>> z.value == pytest.approx(2**3)
        True

        """
        try:
            z = Rev_Var(k**var.value)
            var.children.append((k**var.value * np.log(k), z))
            return z
        except AttributeError: # two real numbers
            return k**var

    @staticmethod
    def logistic(var):
        """ returns a Var as the result of 1 / (1 + e^(-var))

        INPUT
        =======
        var: a Rev_Var object or a real number
        
        RETURNS
        =======
        Rev_Var object: a new Var object with new value and children

        EXAMPLES
        =======
        >>> x = Rev_Var(3.0)
        >>> z = Rev_Var.logistic(x)
        >>> z.grad_value = 1
        >>> x.grad() == pytest.approx(np.exp(3) / ((1 + np.exp(3))**2))
        True
        >>> z.value == pytest.approx(1 / (1+np.exp(-3)))
        True
        """
        try:
            z = Rev_Var(1 / (1 + np.exp(-var.value))) # logistic(x) = 1 / (1 + e^(-x))
            var.children.append((np.exp(var.value) / ((1 + np.exp(var.value))**2), z)) # weight = dz/dvar = e^x/ ((1 + e^x)**2)
            return z
        except AttributeError: # two real numbers
            return 1 / (1 + np.exp(-var))

    @staticmethod
    def sinh(var):
        """ returns a Var as the result of var.sinh()

        INPUT
        =======
        var: a Var object or a real number
        
        RETURNS
        =======
        Rev_Var object: a new Rev_Var object with new value and children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(3.0)
        >>> z1 = Rev_Var.sinh(x)
        >>> print('sinh(x): {}'.format(vars(z1)))
        sinh(x): {'value': 10.017874927409903, 'children': [], 'grad_value': None}
        

        >>> x = Rev_Var(3.0)
        >>> z = Rev_Var.sinh(x)
        >>> z.grad_value = 1
        >>> x.grad() == pytest.approx((np.exp(3)+np.exp(-3)) / 2)
        True
        >>> z.value == pytest.approx((np.exp(3)-np.exp(-3)) / 2)
        True

        """
        try:
            z = Rev_Var((np.exp(var.value) - np.exp(-var.value)) / 2) # sinh(x) = (e^x - e^(-x)) / 2
            var.children.append(((np.exp(var.value)+np.exp(-var.value)) / 2, z)) # weight = dz/dvar = (e^x + e^(-x)) / 2
            return z
        except: # two real numbers
            return (np.exp(var) - np.exp(-var)) / 2

    @staticmethod
    def cosh(var):
        """ returns a Var as the result of var.cosh()

        INPUT
        =======
        var: a Var object or a real number
        
        RETURNS
        =======
        Var object: a new Var object with new value and children

        
        EXAMPLES
        =======
        >>> x = Rev_Var(3.0)
        >>> z1 = Rev_Var.cosh(x)
        >>> print('cosh(x): {}'.format(vars(z1)))
        cosh(x): {'value': 10.067661995777765, 'children': [], 'grad_value': None}

        >>> x = Rev_Var(3.0)
        >>> z = Rev_Var.cosh(x)
        >>> z.grad_value = 1
        >>> x.grad() == pytest.approx((np.exp(3)-np.exp(-3)) / 2)
        True
        >>> z.value == pytest.approx((np.exp(3)+np.exp(-3)) / 2)
        True

        """
        try:
            z = Rev_Var((np.exp(var.value) + np.exp(-var.value)) / 2) # cosh(x) = (e^x + e^(-x)) / 2
            var.children.append(((np.exp(var.value) - np.exp(-var.value)) / 2, z)) # weight = dz/dvar = (e^x - e^(-x)) / 2
            return z
        except: # two real numbers
            return (np.exp(var) + np.exp(-var)) / 2
    
    @staticmethod
    def tanh(var):
        """ returns a Var as the result of var.cosh()

        INPUT
        =======
        var: a Var object or a real number
        
        RETURNS
        =======
        Rev_Var object: a new Rev_Var object with new value and children

        
        EXAMPLES
        =======
        >>> x = Rev_Var(2.0)
        >>> z1 = Rev_Var.sqrt(x)
        >>> z1.grad_value = 1
        >>> print(x.grad(), 0.5*(2**(-0.5)))
        0.3535533905932738 0.3535533905932738
        >>> print(z1.value, np.sqrt(2))
        1.4142135623730951 1.4142135623730951
        
        >>> x = Rev_Var(3.0)
        >>> z = Rev_Var.tanh(x)
        >>> z.grad_value = 1
        >>> x.grad() == pytest.approx(4 / (np.exp(6) + np.exp(-6) + 2))
        True
        >>> z.value == pytest.approx((np.exp(3)-np.exp(-3)) / (np.exp(3)+np.exp(-3)))
        True
        """
        try:
            return Rev_Var.sinh(var) / Rev_Var.cosh(var)
        except: # two real numbers
            return (np.exp(var) - np.exp(-var)) / (np.exp(var) + np.exp(-var))
    
    @staticmethod
    def sqrt(var):
        """ returns a Rev_Var as the result of var.sqrt()

        INPUT
        =======
        var: a Rev_Var object or a real number
        
        RETURNS
        =======
        Rev_Var object: a new Rev_Var object with new value and children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(2.0)
        >>> z1 = Rev_Var.sqrt(x)
        >>> z1.grad_value = 1
        >>> print(x.grad(), 0.5*(2**(-0.5)))
        0.3535533905932738 0.3535533905932738
        >>> print(z1.value, np.sqrt(2))
        1.4142135623730951 1.4142135623730951

        """
        try:
            z = Rev_Var(np.sqrt(var.value))
            var.children.append((1/2 * (var.value**(-1/2)), z))
            return z
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
        Var object: a new Var object with new value and children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(3.0)
        >>> y = Rev_Var(2.0)
        >>> z1 = Rev_Var.sin(x)
        >>> z2 = Rev_Var.sin(y)
        >>> print('sin(x): {}'.format(vars(z1)))
        sin(x): {'value': 0.1411200080598672, 'children': [], 'grad_value': None}
        >>> print('sin(y): {}'.format(vars(z2)))
        sin(y): {'value': 0.9092974268256817, 'children': [], 'grad_value': None}
        """
        
        try:
            z = Rev_Var(np.sin(var.value))
            var.children.append((np.cos(var.value), z)) # weight = dz/dvar = cos(var.value)
            return z
        except:
            return np.sin(var)
        

    @staticmethod
    def cos(var):
        """ returns a Var as the result of var.cos()

        INPUT
        =======
        var: a Var object or a real number
        
        RETURNS
        =======
        Var object: a new Var object with new value and children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(3.0)
        >>> y = Rev_Var(2.0)
        >>> z1 = Rev_Var.cos(x)
        >>> z2 = Rev_Var.cos(y)
        >>> print('cos(x): {}'.format(vars(z1)))
        cos(x): {'value': -0.9899924966004454, 'children': [], 'grad_value': None}
        >>> print('cos(y): {}'.format(vars(z2)))
        cos(y): {'value': -0.4161468365471424, 'children': [], 'grad_value': None}
        
        """
        try:
            z = Rev_Var(np.cos(var.value))
            var.children.append((-np.sin(var.value), z)) # weight = dz/dvar = -sin(var.value)
            return z
        except:
            return np.cos(var)
    
    @staticmethod
    def tan(var):
        """ returns a Var as the result of var.tan()

        INPUT
        =======
        var: a Var object or a real number
        
        RETURNS
        =======
        Var object: a new Var object with new value and children
        
        EXAMPLES
        =======
        >>> x = Rev_Var(3.0)
        >>> y = Rev_Var(2.0)
        >>> z1 = Rev_Var.tan(x)
        >>> z2 = Rev_Var.tan(y)
        >>> print('tan(x): {}'.format(vars(z1)))
        tan(x): {'value': -0.1425465430742778, 'children': [], 'grad_value': None}
        >>> print('tan(y): {}'.format(vars(z2)))
        tan(y): {'value': -2.185039863261519, 'children': [], 'grad_value': None}
        """
        
        try:
            z = Rev_Var(np.tan(var.value))
            var.children.append((1 / (np.cos(var.value) ** 2), z)) # weight = dz/dvar = 1/(np.cos(var.value)^2)
            return z
        except:
            return np.tan(var)
    @staticmethod
    def arcsin(var):
        """ returns a Rev_Var as the result of var.arcsin()

        INPUT
        =======
        var: a Rev_Var object or a real number
        
        RETURNS
        =======
        Rev_Var object: a new Rev_ Var object with new value and children
        
        EXAMPLES
        =======
        """
        
        try:
            z = Rev_Var(np.arcsin(var.value))
            var.children.append((1 / ((1 - var.value ** 2) ** 0.5), z)) # weight = dz/dvar
            return z
        except:
            return np.arcsin(var)


    @staticmethod
    def arccos(var):
        """ returns a Rev_Var as the result of var.arccos()

        INPUT
        =======
        var: a Rev_Var object or a real number
        
        RETURNS
        =======
        Rev_Var object: a new Rev_ Var object with new value and children
        
        EXAMPLES
        =======
        """
        
        try:
            z = Rev_Var(np.arccos(var.value))
            var.children.append((-1 / ((1 - var.value ** 2) ** 0.5), z)) # weight = dz/dvar
            return z
        except:
            return np.arccos(var)
    
    @staticmethod
    def arctan(var):
        """ returns a Rev_Var as the result of var.arctan()

        INPUT
        =======
        var: a Rev_Var object or a real number
        
        RETURNS
        =======
        Rev_Var object: a new Rev_ Var object with new value and children
        
        EXAMPLES
        =======
        """
    
        try:
            z = Rev_Var(np.arctan(var.value))
            var.children.append((1 / (1 + var.value ** 2), z)) # weight = dz/dvar
            return z
        except:
            return np.arctan(var)
    
    
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

    # x = Rev_Var(0.5)
    # y = Rev_Var(4.2)
    # z = x * y + Rev_Var.sin(x)
    # z.grad_value = 1.0

    # assert z.value == pytest.approx(2.579425538604203)
    # assert x.grad() == pytest.approx(y.value + np.cos(x.value))
    # assert y.grad() == pytest.approx(x.value)

    # x = Rev_Var(5)
    # y = Rev_Var(1)
    # z = Rev_Var.log(x) ** Rev_Var.sin(y)
    # z.grad_value = 1.0
    # print(z.value, x.grad(), y.grad())

    # # sinh
    # y = Rev_Var(3.0)
    # z = Rev_Var.sinh(y)
    # z.grad_value = 1
    # print(y.grad(), (np.exp(3) + np.exp(-3))/2)

    # # expk
    # y = Rev_Var(3.0)
    # z = Rev_Var.expk(4.0, y)
    # z.grad_value = 1
    # print(y.grad(), 4**3*np.log(4))


    # # pow
    # z1 = x**y
    # # set final derivative df/dx_final = 1
    # z1.grad_value = 1
    # print(vars(z1))
    # # calculate final partial derivative df/dx and df/dy
    # print(x.grad(), 3*(2**2))
    # print(vars(x))
    # print(y.grad(), 2**3*np.log(2))
    # print(vars(y))

    # y = Rev_Var(3.0)
    # z2 = y**3
    # # set final derivative df/dx_final = 1
    # z2.grad_value = 1
    # print(vars(z2))
    # # calculate final partial derivative df/dx and df/dy
    # print(y.grad(), 3*(3**2))
    # print(vars(y))


    # # rpow
    # z3 = 2**y
    # z3.grad_value = 1
    # print(vars(z3))
    # print(y.grad(), 2**3*np.log(2))
    # print(vars(y))

    # # div
    # z4 = x/y
    # z4.grad_value = 1
    # print(x.grad(), 1/3)
    # print(y.grad(),-2/(3**2))


    # z5 = x/2
    # z5.grad_value = 1
    # print(x.grad(), 1/2)

    # z6 = 2/y
    # z6.grad_value = 1
    # print(y.grad(), -2/(3**2))


    # # neg
    # z5 = x/2
    # z5.grad_value = 1
    # x.grad()
    # print(vars(x))
    # print(vars(-x))

    # print(vars(-y))


    # # eq
    # z6 = x/2
    # z7 = x/2
    # print(z6 == z7)

    # x2 = Rev_Var(2.0)
    # print(x == x2)

    # # neq
    # z6 = x/2
    # z7 = x/2
    # print(z6 != z7)

    # x = Rev_Var(2.0)
    # x2 = Rev_Var(2.0)
    # print(x != x2)
    # y = Rev_Var(3.0)
    # print(x != y)


    # # log
    # x = Rev_Var(2.0)
    # z1 = Rev_Var.log(x)
    # z1.grad_value = 1
    # print(x.grad(), 1/2)

    # # logk
    # x = Rev_Var(2.0)
    # z2 = Rev_Var.logk(x, 3.0)
    # z2.grad_value = 1
    # print(z2.value, np.log(2) / np.log(3))
    # print(x.grad(), 1/(2*np.log(3)))


    # # exp
    # x = Rev_Var(2.0)
    # z1 = Rev_Var.exp(x)
    # z1.grad_value = 1
    # print(x.grad(), np.exp(2))
    # print(z1.value, np.exp(2))

    # # sqrt
    # x = Rev_Var(2.0)
    # z1 = Rev_Var.sqrt(x)
    # z1.grad_value = 1
    # print(x.grad(), 0.5*(2**(-0.5)))
    # print(z1.value, np.sqrt(2))

