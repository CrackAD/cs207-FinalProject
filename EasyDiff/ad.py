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
        >>> ad = AD(np.array([2, 2]), np.array([1, 1]))
        >>> print(vars(ad.vars[0]), vars(ad.vars[1]))
        {'val': 2, 'der': array([1, 0])} {'val': 2, 'der': array([0, 1])}
        """
        assert(len(vals) == len(ders))

        self.vars = []
        dimen = len(vals)
        cnt = 0
        for val, der in zip(vals, ders):
            der_list = np.array([0 for i in range(dimen)])
            der_list[cnt] = der
            self.vars.append(Var(val, der_list))
            cnt += 1

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
        return func(*self.vars)

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
            res_der = self.auto_diff(func).der
            for j in range(len(self.vars)):
                res[i][j] = res_der[j]
        return res       


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)


    # f1 = lambda x, y: Var.log(x) ** Var.sin(y)
    # ad = AD(np.array([2, 2]), np.array([1, 1]))
    # print("Var.log(x) ** Var.sin(y): {}".format(vars(ad.auto_diff(f1))))

    # f1 = lambda x: Var.log(x) ** 2
    # ad = AD(np.array([2]), np.array([1]))
    # print("Var.log(x) ** 2: {}".format(vars(ad.auto_diff(f1))))

    # f1 = lambda x, y: Var.log(x) ** Var.sin(y)
    # f2 = lambda x, y: Var.sqrt(x) / y
    # ad = AD(np.array([4.12, 5.13]), np.array([1, 1]))
    # print("jac_matrix: \n{}".format(ad.jac_matrix([f1, f2])))

