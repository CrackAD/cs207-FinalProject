# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 10:33:10 2019

@author: Emily Gould
"""
import numpy as np
import pytest
from rev_var import Var

#a = Var(1)
#print(a.value, a.children, a.grad_value)

#a = Var(1)
#print(Var.grad(a))

#added = Var.__add__(a, 4)
#print(added.val)

x = 4
y = 3
z1 = x**y
# set final derivative df/dx_final = 1
z1.grad_value = 1
print(vars(z1))
# calculate final partial derivative df/dx and df/dy
print(x.grad(), 3*(2**2))
print(vars(x))
print(y.grad(), 2**3*np.log(2))
print(vars(y))
