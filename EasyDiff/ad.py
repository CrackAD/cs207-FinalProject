from var import Var
import numpy as np

class AD():
    def __init__(self, vals, ders):
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
        >>> f1 = lambda x, y: Var.log(x) ** Var.sin(y)
    
        >>> ad = AD(np.array([2, 2]), np.array([1, 1]))
        >>> print("Var.log(x) ** Var.sin(y): {}".format(vars(ad.auto_diff(f1))))
        Var.log(x) ** Var.sin(y): {'val': 0.7165772257590739, 'der': array([0.47001694, 0.10929465])}
        """
        assert(len(vals) == len(ders))

        self.vars = []
        self.dimen = len(vals)
        cnt = 0
        for val, der in zip(vals, ders):
            der_list = [0 for i in range(self.dimen)]
            der_list[cnt] = der
            self.vars.append(Var(val, der_list))
            cnt += 1

    def auto_diff(self, func):
        """
        passing all of the items in the fruits list into the print function call as separate arguments, 
        without us even needing to know how many arguments are in the list.

        INPUT
        =======
        a function
        
        RETURNS
        =======
        returns the attributes of the function
        
        EXAMPLES
        =======
        >>> f1 = lambda x: Var.log(x) ** 2
        
        >>> ad = AD(np.array([2]), np.array([1]))
        >>> print("Var.log(x) ** 2: {}".format(vars(ad.auto_diff(f1))))
        Var.log(x) ** 2: {'val': 0.4804530139182014, 'der': array([0.69314718])}        
        """
        return func(*self.vars)

    def jac_matrix(self, funcs):
        pass
        """
        seems like we need to finalize

        INPUT
        =======
        lorem ipsum
        
        RETURNS
        =======
        lorem ipsum
        
        EXAMPLES
        =======
        """

if __name__ == "__main__":
    f1 = lambda x, y: Var.log(x) ** Var.sin(y)
    
    ad = AD(np.array([2, 2]), np.array([1, 1]))
    print("Var.log(x) ** Var.sin(y): {}".format(vars(ad.auto_diff(f1))))

    f1 = lambda x: Var.log(x) ** 2
    
    ad = AD(np.array([2]), np.array([1]))
    print("Var.log(x) ** 2: {}".format(vars(ad.auto_diff(f1))))
    

