from EasyDiff.ad import AD
from EasyDiff.var import Var
import numpy as np

# give it a function of your choice
func = lambda x,y: Var.log(x) ** Var.sin(y)

# give the initial values to take the derivatives at
ad = AD(vals=np.array([2, 2]), ders=np.array([1, 1]))

# calculate and print the derivatives
print("Var.log(x) ** Var.sin(y): {}".format(vars(ad.auto_diff(func))))