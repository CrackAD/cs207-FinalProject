import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from EasyDiff.var import Var
from EasyDiff.rev_var import Rev_Var
import numpy as np
from enum import Enum

class AD_Mode:
    FORWARD = 0
    REVERSE = 1

class AD():
    def __init__(self, vals, ders, mode):
        """
        init the AD class
        INPUT
        =======
        self: an AD class
        vals: a list of initial value for a list of variables
        ders: a list of initial derivative for a list of variables
        
        RETURNS
        =======
        the output of automatic differentiation for the given vals and ders
        
        EXAMPLES
        =======
        >>> ad = AD(np.array([2, 2]), np.array([1, 1]))
        >>> print(vars(ad.vars[0]), vars(ad.vars[1]))
        {'val': 2, 'der': array([1, 0])} {'val': 2, 'der': array([0, 1])}
        """
        self.mode = mode
        if self.mode == AD_Mode.FORWARD:
            assert(len(vals) == len(ders))

            self.vars = []
            dimen = len(vals)
            cnt = 0
            for val, der in zip(vals, ders):
                der_list = np.array([0 for i in range(dimen)])
                der_list[cnt] = der
                self.vars.append(Var(val, der_list))
                cnt += 1
        else:
            self.vals = vals
            self.vars = []
            for val in self.vals:
                self.vars.append(Rev_Var(val))
            
    def clear(self):
        if self.mode == AD_Mode.REVERSE:
            self.vars = []
            for val in self.vals:
                self.vars.append(Rev_Var(val))

    def auto_diff(self, func):
        """
        Passing a function to a AD object, and return the final Var object with val and der.

        INPUT
        =======
        a function
        
        RETURNS
        =======
        returns the final Var object with val and der
        
        EXAMPLES
        =======
        >>> f1 = lambda x, y: Var.log(x) ** Var.sin(y)
        >>> ad = AD(np.array([2, 2]), np.array([1, 1]))
        >>> print("Var.log(x) ** Var.sin(y): {}".format(vars(ad.auto_diff(f1))))
        Var.log(x) ** Var.sin(y): {'val': 0.7165772257590739, 'der': array([0.47001694, 0.10929465])}
        >>> f1 = lambda x: Var.log(x) ** 2
        >>> ad = AD(np.array([2]), np.array([1]))
        >>> print("Var.log(x) ** 2: {}".format(vars(ad.auto_diff(f1))))
        Var.log(x) ** 2: {'val': 0.4804530139182014, 'der': array([0.69314718])}
        """
        if self.mode == AD_Mode.FORWARD:
            return func(*self.vars)
        else:
            self.clear()
            z = func(*self.vars)
            z.grad_value = 1.0
            res = list(map(lambda x: x.grad(), self.vars))
            return Var(z.value, np.array(res)) # provide a unify interface

    def jac_matrix(self, funcs):
        pass
        """
        Passing a list of functions to a AD object, and return the Jacobian Matrix.

        INPUT
        =======
        a list of functions
        
        RETURNS
        =======
        Jacobian Matrix

        EXAMPLES
        =======
        >>> f1 = lambda x, y: Var.log(x) ** Var.sin(y)
        >>> f2 = lambda x, y: Var.sqrt(x) / y
        >>> ad = AD(np.array([4.12, 5.13]), np.array([1, 1]))
        >>> print("jac_matrix: \n{}".format(ad.jac_matrix([f1, f2])))
        jac_matrix: 
        [[-0.11403015  0.10263124]
         [ 0.048018   -0.07712832]]
        """
        res = np.zeros(shape=(len(funcs), len(self.vars)))
        for i, func in enumerate(funcs):
            if self.mode == AD_Mode.REVERSE:
                self.clear()
            res_der = self.auto_diff(func).der
            for j in range(len(self.vars)):
                res[i][j] = res_der[j]
        return res

if __name__ == "__main__":
    # import doctest
    # doctest.testmod(verbose=True)


    f1 = lambda x, y: Var.log(x) ** Var.sin(y)
    ad = AD(np.array([5, 1]), np.array([1, 1]), AD_Mode.FORWARD)
    print("Var.log(x) ** Var.sin(y): {}".format(vars(ad.auto_diff(f1))))

    f1 = lambda x, y: Rev_Var.log(x) ** Rev_Var.sin(y)
    ad = AD(np.array([5, 1]), np.array([1, 1]), AD_Mode.REVERSE)
    print("Var.log(x) ** Var.sin(y): {}".format(vars(ad.auto_diff(f1))))



    f1 = lambda x: Var.log(x) ** 2
    ad = AD(np.array([2]), np.array([1]), AD_Mode.FORWARD)
    print("Var.log(x) ** 2: {}".format(vars(ad.auto_diff(f1))))

    f1 = lambda x: Rev_Var.log(x) ** 2
    ad = AD(np.array([2]), np.array([1]), AD_Mode.REVERSE)
    print("Var.log(x) ** 2: {}".format(vars(ad.auto_diff(f1))))



    f1 = lambda x, y: Var.log(x) ** Var.sin(y)
    f2 = lambda x, y: Var.sqrt(x) / y
    ad = AD(np.array([4.12, 5.13]), np.array([1, 1]), AD_Mode.FORWARD)
    print("jac_matrix: \n{}".format(ad.jac_matrix([f1, f2])))

    f1 = lambda x, y: Rev_Var.log(x) ** Rev_Var.sin(y)
    f2 = lambda x, y: Rev_Var.sqrt(x) / y
    ad = AD(np.array([4.12, 5.13]), np.array([1, 1]), AD_Mode.REVERSE)
    print("jac_matrix: \n{}".format(ad.jac_matrix([f1, f2])))

