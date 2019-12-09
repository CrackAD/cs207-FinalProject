from EasyDiff.ad import AD
from EasyDiff.var import Var
from EasyDiff.rev_var import Rev_Var
from EasyDiff.ad import AD_Mode
import numpy as np

# test forward mode. 
# give it a function of your choice
func = lambda x,y: Var.log(x) ** Var.sin(y)

# give the initial values to take the derivatives at
ad = AD(vals=np.array([2, 2]), ders=np.array([1, 1]), mode=AD_Mode.FORWARD)

# calculate and print the derivatives
print("Var.log(x) ** Var.sin(y): {}".format(vars(ad.auto_diff(func))))

# test reverse mode. 
func = lambda x,y: Rev_Var.log(x) ** Rev_Var.sin(y)
ad = AD(vals=np.array([2, 2]), ders=np.array([1, 1]), mode=AD_Mode.REVERSE)
print("Rev_Var.log(x) ** Rev_Var.sin(y): {}".format(vars(ad.auto_diff(func))))
