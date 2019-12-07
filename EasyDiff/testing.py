# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 10:33:10 2019

@author: Emily Gould
"""
import numpy as np
import pytest
from rev_var import Rev_Var

print(y)

#a = Var(1)
#print(a.value, a.children, a.grad_value)

#a = Var(1)
#print(Var.grad(a))

#added = Var.__add__(a, 4)
#print(added.val)

#x = 4
#y = 3
#z1 = x**y
## set final derivative df/dx_final = 1
#z1.grad_value = 1
#print(vars(z1))
## calculate final partial derivative df/dx and df/dy
#print(x.grad(), 3*(2**2))
#print(vars(x))
#print(y.grad(), 2**3*np.log(2))
#print(vars(y))

#
#y = Rev_Var(3.0)
#z = Rev_Var.expk(4.0, y)
#z.grad_value = 1
#print(y.grad(), 4**3*np.log(4))
#
#x = Rev_Var(2.0)
#z1 = Rev_Var.sqrt(x)
#z1.grad_value = 1
#print(x.grad(), 0.5*(2**(-0.5)))
#print(z1.value, np.sqrt(2))

'''
not working
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

'''
# rpow
z3 = 2**y
z3.grad_value = 1
print(vars(z3))
print(y.grad(), 2**3*np.log(2))
print(y.value, y.children)

'''
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

'''
