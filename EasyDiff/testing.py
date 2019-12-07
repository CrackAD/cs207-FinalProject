# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 10:33:10 2019

@author: Emily Gould
"""
import numpy as np
import pytest
from rev_var import Rev_Var

#y = Rev_Var(3.0)
#z = Rev_Var.expk(4.0, y)
#z.grad_value = 1
#print(y.grad(), 4**3*np.log(4))

#a = Rev_Var(1)
#print(a.value, a.children, a.grad_value)

#a = Rev_Var(1)
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
#
#print(z1.value, np.sqrt(2))


'''
# pow
y = Rev_Var(3.0)
z1 = x**y
# set final derivative df/dx_final = 1
z1.grad_value = 1
print(vars(z1))
 calculate final partial derivative df/dx and df/dy
print(x.grad(), 3*(2**2))
print(vars(x))
print(y.grad(), 2**3*np.log(2))
print(vars(y))


y = Rev_Var(3.0)
z2 = y**3
# set final derivative df/dx_final = 1
z2.grad_value = 1
print(vars(z2))
# calculate final partial derivative df/dx and df/dy
print(y.grad(), 3*(3**2))
print(vars(y))
'''
#y = Rev_Var(3.0)
#x = Rev_Var(2.0)
#z4 = x/y
#z4.grad_value = 1
#print(x.grad(), 1/3)
#print(y.grad(),-2/(3**2))



## rpow
#z3 = 2**y
#z3.grad_value = 1
#print(vars(z3))
#print(y.grad(), 2**3*np.log(2))
#print(y.value, y.children)


# div
#y = Rev_Var(3.0)
#z4 = x/y
#z4.grad_value = 1
#print(x.grad(), 1/3)
#0.3535533905932738 0.3333333333333333
#print(y.grad(),-2/(3**2))
#26.77777777777778 -0.2222222222222222
'''

z5 = x/2
z5.grad_value = 1
print(x.grad(), 1/2)

z6 = 2/y
z6.grad_value = 1
print(y.grad(), -2/(3**2))

'''
#This one is wonky
## neg
#x = Rev_Var(2.0)
#y = Rev_Var(3.0)
#z5 = x/2
#z5.grad_value = 1
#x.grad()
#print(vars(x))
#print(vars(-x))
#print(vars(-y))
'''
## eq
#z6 = x/2
#z7 = x/2
#print(z6 == z7)
#
#x2 = Rev_Var(2.0)
#print(x == x2)
'''
## neq
#z6 = x/2
#z7 = x/2
#print(z6 != z7)
#
#x = Rev_Var(2.0)
#x2 = Rev_Var(2.0)
#print(x != x2)
#y = Rev_Var(3.0)
#print(x != y)
'''
'''
'''
# log
x = Rev_Var(2.0)
z1 = Rev_Var.log(x)
z1.grad_value = 1
print(x.grad(), 1/2)


# logk
x = Rev_Var(2.0)
z2 = Rev_Var.logk(x, 3.0)
z2.grad_value = 1
print(z2.value, np.log(2) / np.log(3))
print(x.grad(), 1/(2*np.log(3)))
'''
## exp
#x = Rev_Var(2.0)
#z1 = Rev_Var.exp(x)
#z1.grad_value = 1
#print(x.grad(), np.exp(2))
#print(z1.value, np.exp(2))


#x = Rev_Var(2.0)
#y = Rev_Var(3.0)    
#z1 = x + y
#print('x + y: {}'.format(vars(z1)))
#z2 = x + 1
#print('x + 1: {}'.format(vars(z2)))
#z3 = 1 + x
#print('1 + x: {}'.format(vars(z3)))

#x = Rev_Var(2.0)
#t = 3 + x
#print(t.value, t.grad_value)


#x = Rev_Var(3.0)
#y = Rev_Var(2.0)
#z = Rev_Var(3.0)
#z4 = y*2
#z5 = 2*y
#z6 = -1*y
#z7 = y*(-1)
#z8 = x*y
#print(vars(z4))
#print(vars(z5))
#print(vars(z6))
#print(vars(z7)) 
#print(vars(z8))  

#
#x = Rev_Var(2.0)
#t = 3 * x
#print(t.value, t.grad_value)
#        
#
#x = Rev_Var(3.0)
#y = Rev_Var(2.0)
#z = Rev_Var(3.0)
#z1 = x - y
#print('x - y: {}'.format(vars(z1)))
#
#z2 = x - 2
#print('x - 2: {}'.format(vars(z2)))
#
#z3 = 2 - x
#print('2 - x: {}'.format(vars(z3)))

#
#x = Rev_Var(2.0)
#t = 3 - x
#print(t.value, t.grad_value)

#x = Rev_Var(3.0)
#y = Rev_Var(2.0)
#z = Rev_Var(3.0)
#z1 = x**y
#print('x ** y: {}'.format(vars(z1)))
#
#z2 = x**2
#print('x ** 2: {}'.format(vars(z2)))

#x = Rev_Var(3.0)
#y = Rev_Var(2.0)
#z = Rev_Var(3.0)
#z1 = x**y
#print('x ** y: {}'.format(vars(z1)))
#
#z2 = x**2
#print('x ** 2: {}'.format(vars(z2)))
#
#z3 = 2**x
#print('2 ** x: {}'.format(vars(z3)))
#
#z4 = x**(-1)
#print('x ** (-1): {}'.format(vars(z4)))
#
#x = Rev_Var(3.0)
#y = Rev_Var(2.0)
#p = x * (y * (-1))
#print(p.value, p.grad_value)

#x = Rev_Var(3.0)
#y = Rev_Var(2.0)
#print(x.value, x.grad_value)
#print(y.value, y.grad_value)

#x = Rev_Var(3.0)
#y = Rev_Var(2.0)
#z = Rev_Var(3.0)
#print(x!=y)
#
#print(x != z)


#x = Rev_Var(3.0)
#y = Rev_Var(2.0)
#z = Rev_Var(3.0)
#z1 = Rev_Var.exp(x)
#z2 = Rev_Var.exp(y)
#print('exp(x): {}'.format(vars(z1)))
#print('exp(y): {}'.format(vars(z2)))

#x = Rev_Var(3.0)
#z1 = Rev_Var.sinh(x)
#print('sinh(x): {}'.format(vars(z1)))
#
#x = Rev_Var(3.0)
#z1 = Rev_Var.cos(x)
#print('cos(x): {}'.format(vars(z1)))

#x = Rev_Var(3.0)
#z1 = Rev_Var.tan(x)
#print('tan(x): {}'.format(vars(z1)))

#
#x = Rev_Var(3.0)
#y = Rev_Var(2.0)
#z1 = Rev_Var.sin(x)
#z2 = Rev_Var.sin(y)
#print('sin(x): {}'.format(vars(z1)))
#
#print('sin(y): {}'.format(vars(z2)))

#x = Rev_Var(3.0)
#y = Rev_Var(2.0)
#z1 = Rev_Var.cos(x)
#z2 = Rev_Var.cos(y)
#print('cos(x): {}'.format(vars(z1)))
#
#print('cos(y): {}'.format(vars(z2)))


#x = Rev_Var(3.0)
#y = Rev_Var(2.0)
#z1 = Rev_Var.tan(x)
#z2 = Rev_Var.tan(y)
#print('tan(x): {}'.format(vars(z1)))
#
#print('tan(y): {}'.format(vars(z2)))

#x = Rev_Var(3.0)
#z1 = Rev_Var.cosh(x)
#print('cosh(x): {}'.format(vars(z1)))
#cosh(x): {'value': 10.067661995777765, 'children': [], 'grad_value': None}
#
#x = Rev_Var(3.0)
#z = Rev_Var.cosh(x)
#z.grad_value = 1
#x.grad() == pytest.approx((np.exp(3)-np.exp(-3)) / 2)
#
#z.value == pytest.approx((np.exp(3)+np.exp(-3)) / 2)


x = Rev_Var(3.0)
z = Rev_Var.arcsin(x)
z.grad_value = 1
x.grad() == pytest.approx((np.exp(3)-np.exp(-3)) / 2)

z.value == pytest.approx((np.exp(3)+np.exp(-3)) / 2)
