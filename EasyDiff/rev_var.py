import numpy as np
import pytest
class Rev_Var():
    '''
    This class defines a multivariate reverse mode node
    '''
    def __init__(self, value):
        """ constructor for Var class

        INPUT
        =======
        value: value of the current variable
        
        RETURNS
        =======
        Var object: self.value, self.children [(weight, Var)], and self.grad_value
        
        EXAMPLES
        =======
        >>> a = Rev_Var(1)
        >>> print(a.value, a.children, a.grad_value)
        1 [] None
        
        """
        self.value = value
        self.children = [] # store the <weight, Var> tuple for all its child. 
        self.grad_value = None # store derivatives of final expression with respect to current variable

    def grad(self):
        """ calculate partial derivatives of the final expression with respect to current variable

        INPUT
        =======
        self: a Var object 
        
        RETURNS
        =======
        Var object: a Var object with updated partial derivative df/dself
        
        EXAMPLES
        =======
        >>> y = Rev_Var(3.0)
        >>> z = Rev_Var.expk(4.0, y)
        >>> z.grad_value = 1
        >>> print(y.grad(), 4**3*np.log(4))
        88.722839111673 88.722839111673
        """
        if self.grad_value is None:
            self.grad_value = sum(weight * var.grad()
                                  for weight, var in self.children)
        return self.grad_value

    def __add__(self, other):
        """ returns a Var as the result of self + other

        INPUT
        =======
        self: a Var object (object before +)
        other: a Var object or a real number (object after +)
        
        RETURNS
        =======
        Var object: a new Var object with new value and no children
        
        EXAMPLES
        =======
        
        
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
        """ return a Var as the result of other + self

        INPUT
        =======
        self: a Var object (object after +)
        other: a Var object or a real number (object before +)
        
        RETURNS
        =======
        Var object: a new Var object with new value and no children
        
        EXAMPLES
        =======
        
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
        Var object: a new Var object with value and no children
        
        EXAMPLES
        =======
        
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
        """ returns a Var as the result of other * self
        
        INPUT
        =======
        self: a Var object (object after *)
        other: a Var object or a real number (object before *)
        
        RETURNS
        =======
        Var object: a new Var object with new value and no children
        
        EXAMPLES
        =======
        
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
        Var object: a new Var object with new value and children
        
        EXAMPLES
        =======
        
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
        """ returns a Var as the result of other - self
        
        INPUT
        =======
        self: a Var object (object after -)
        other: a Var object or a real number (object before -)
            
        RETURNS
        =======
        Var object: a new Var object with new value and children
        
        EXAMPLES
        =======
        
        """
        return -1 *(self - other)

    def __pow__(self, other):
        """" returns a Var as the result of self**(other)
        
        INPUT
        =======
        self: a Var object (object before **)
        other: a Var object or a real number (object after **)
        
        RETURNS
        =======
        
        EXAMPLES
        =======
        
        Var object: a new Var object with new value and children
        """
        try: # two Var objects
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
        """ returns a Var as the result of other**(self)

        INPUT
        =======
        self: a Var object (object after **)
        other: a Var object or a real number (object before **)
        
        RETURNS
        =======
        Var object: a new Var object with new value and children
        
        EXAMPLES
        =======
        >>> y = Rev_Var(3.0)
        >>> z3 = 2**y
        >>> z3.grad_value = 1
        >>> print(y.grad(), 2**3*np.log(2))
        5.545177444479562 5.545177444479562
        
        """
        # the only scenario using this is when other is a real number and self is a Var object
        z = Rev_Var(other **self.value)
        self.children.append(((other**self.value) * np.log(other), z))
        return z

    def __truediv__(self, other):
        """ returns a Var as the result of self / other

        INPUT
        =======
        self: a Var object (numerator)
        other: a Var object or a real number (denominator)
        
        RETURNS
        =======
        Var object: a new Var object with new value and children
        
        EXAMPLES
        =======
        
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
        Var object: a new Var object with new value and children
        
        EXAMPLES
        =======
        
        """
        return other*(self**(-1))
    
    def __neg__(self):
        """ returns a Var as the result of - self

        INPUT
        =======
        self: a Var object
        
        RETURNS
        =======
        Var object: a new Var object with new value and children
        
        EXAMPLES
        =======
        
        """
        ans= Rev_Var(-self.value)
        ans.children = [(-1, self)]
        if self.grad_value:
            ans.grad_value = -self.grad_value
        return ans

    
    def __pos__(self):
        """ returns a Var as the result of + self

        INPUT
        =======
        self: a Var object
        
        RETURNS
        =======
        Var object: a Var object
        
        EXAMPLES
        =======
        
        """
        return self.copy()

    def __eq__(self, other):
        """ returns the result of self == other

        INPUT
        =======
        self: a Var object (before ==)
        other: a Var object or something else(after ==)
        
        RETURNS
        =======
        a boolean value
        
        EXAMPLES
        =======
        
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
        self: a Var object (before !=)
        other: a Var object or something else(after !=)
        
        RETURNS
        =======
        a boolean value
        
        EXAMPLES
        =======
        
        """
        try:
            return not (self == other)
        except AttributeError:
            return True

    @staticmethod
    def log(var): # ln()
        """ returns a Var as the result of var.log()

        INPUT
        =======
        var: a Var object or real number
        
        RETURNS
        =======
        Var object: a new Var object with new val and children

        EXAMPLES
        =======        
    
        """
        try: # a Var object
            z = Rev_Var(np.log(var.value))
            var.children.append((1/ var.value, z))
            return z
        except AttributeError: # a real number
            return np.log(var)

    @staticmethod
    def logk(var, k): #logk(var)
        """ returns a Var as the result of var.log(k)

        INPUT
        =======
        var: a Var object or real number
        
        RETURNS
        =======
        Var object: a new Var object with new val and children
    
        EXAMPLES
        =======
        
        """
        try: # a Var object
            z = Rev_Var(np.log(var.value) / np.log(k))
            var.children.append((1/ (var.value*np.log(k)), z))
            return z
        except AttributeError: # a real number
            return np.log(var) / np.log(k)
    
    @staticmethod
    def exp(var): #e^(var)
        """ returns a Var as the result of var.exp()

        INPUT
        =======
        var: a Var object or a real number
        
        RETURNS
        =======
        Var object: a new Var object with new value and children
        
        EXAMPLES
        =======

        """
        try:
            z = Rev_Var(np.exp(var.value))
            var.children.append((np.exp(var.value), z))
            return z
        except AttributeError: # two real numbers
            return np.exp(var)
    
    @staticmethod
    def expk(k, var): #k^(var)
        """ returns a Var as the result of var.expk()

        INPUT
        =======
        var: a Var object or a real number
        k: the base of the exponential
        
        RETURNS
        =======
        Var object: a new Var object with new value and children
        """
        try:
            z = Rev_Var(k**var.value)
            var.children.append((k**var.value * np.log(k), z))
            return z
        except AttributeError: # two real numbers
            return k**var

    @staticmethod
    def sinh(var):
        """ returns a Var as the result of var.sinh()

        INPUT
        =======
        var: a Var object or a real number
        
        RETURNS
        =======
        Var object: a new Var object with new value and children
        """
        try:
            z = Rev_Var((np.exp(var.value) - np.exp(-var.value)) / 2)
            var.children.append(((np.exp(var.value) - np.exp(-var.value)) / 2, z)) # weight = dz/dvar = (e^x + e^(-x)) / 2
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
        """
        try:
            z = Rev_Var((np.exp(var.value) - np.exp(-var.value)) / 2)
            var.children.append(((np.exp(var.value) - np.exp(-var.value)) / 2, z)) # weight = dz/dvar = (e^x + e^(-x)) / 2
            return z
        except: # two real numbers
            return (np.exp(var) - np.exp(-var)) / 2
    
    @staticmethod
    def sqrt(var):
        """ returns a Var as the result of var.sqrt()

        INPUT
        =======
        var: a Var object or a real number
        
        RETURNS
        =======
        Var object: a new Var object with new value and children
        
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
        try:
            z = Rev_Var(np.sin(var.value))
            var.children.append((np.cos(var.value), z)) # weight = dz/dvar = cos(var.value)
            return z
        except:
            return np.sin(var)

    @staticmethod
    def cos(var):
        try:
            z = Rev_Var(np.cos(var.value))
            var.children.append((-np.sin(var.value), z)) # weight = dz/dvar = -sin(var.value)
            return z
        except:
            return np.cos(var)
    
    @staticmethod
    def tan(var):
        try:
            z = Rev_Var(np.tan(var.value))
            var.children.append((1 / (np.cos(var.value) ** 2), z)) # weight = dz/dvar = 1/(np.cos(var.value)^2)
            return z
        except:
            return np.tan(var)
        
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    
'''
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
'''


'''
# sinh
y = Rev_Var(3.0)
z = Rev_Var.sinh(y)
z.grad_value = 1
print(y.grad(), (np.exp(3) + np.exp(-3))/2)

# expk
y = Rev_Var(3.0)
z = Rev_Var.expk(4.0, y)
z.grad_value = 1
print(y.grad(), 4**3*np.log(4))
>>>>>>> final-ruby


# pow
z1 = x**y
# set final derivative df/dx_final = 1
z1.grad_value = 1
print(vars(z1))
# calculate final partial derivative df/dx and df/dy
print(x.grad(), 3*(2**2))
print(vars(x))
print(y.grad(), 2**3*np.log(2))
print(vars(y))

y = Var(3.0)
z2 = y**3
# set final derivative df/dx_final = 1
z2.grad_value = 1
print(vars(z2))
# calculate final partial derivative df/dx and df/dy
print(y.grad(), 3*(3**2))
print(vars(y))


# rpow
z3 = 2**y
z3.grad_value = 1
print(vars(z3))
print(y.grad(), 2**3*np.log(2))
print(vars(y))

# div
z4 = x/y
z4.grad_value = 1
print(x.grad(), 1/3)
print(y.grad(),-2/(3**2))


z5 = x/2
z5.grad_value = 1
print(x.grad(), 1/2)

z6 = 2/y
z6.grad_value = 1
print(y.grad(), -2/(3**2))


# neg
z5 = x/2
z5.grad_value = 1
x.grad()
print(vars(x))
print(vars(-x))

print(vars(-y))


# eq
z6 = x/2
z7 = x/2
print(z6 == z7)

x2 = Var(2.0)
print(x == x2)

# neq
z6 = x/2
z7 = x/2
print(z6 != z7)

x = Var(2.0)
x2 = Var(2.0)
print(x != x2)
y = Var(3.0)
print(x != y)


# log
x = Var(2.0)
z1 = Var.log(x)
z1.grad_value = 1
print(x.grad(), 1/2)

# logk
x = Var(2.0)
z2 = Var.logk(x, 3.0)
z2.grad_value = 1
print(z2.value, np.log(2) / np.log(3))
print(x.grad(), 1/(2*np.log(3)))


# exp
x = Var(2.0)
z1 = Var.exp(x)
z1.grad_value = 1
print(x.grad(), np.exp(2))
print(z1.value, np.exp(2))

# sqrt
x = Var(2.0)
z1 = Var.sqrt(x)
z1.grad_value = 1
print(x.grad(), 0.5*(2**(-0.5)))
print(z1.value, np.sqrt(2))
'''

