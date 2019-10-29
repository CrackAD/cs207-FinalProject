## Introduction

## Background

## How to Use EasyDiff
```python
from EasyDiff import Var
from EasyDiff import AD

K = 2 # two input variables
f1 = lambda x,y: x**2 + y
x = Var(K, [2, 1]) # evaluating at x=2 with initial derivative of 1
y = Var(K, [5, 1]) # evaluating at y=2 with initial derivative of 1

res = AD().auto_diff(f1, [x, y]) # get the result Var instance

print("function value: {}".format(res.dual_paras[0]))
print("derivative with respect to x: {}".format(res.dual_paras[1]))
print("derivative with respect to y: {}".format(res.dual_paras[2]))

```
## Software Organization

Our directory structure will look like:

```
EasyDiff/
		easydiff/
				__init__.py
				ad.py
				var.py
				tests/
					__init__.py
					test.py
		README.md
		setup.py
		LICENSE
```
In the directory, we have two python modules `ad.py` and `var.py`. 

* `ad.py`: all automatic differentiation calculations
* `var.py`: dual number constructions and basic math operations overloaded

We also plan on including dependencies `numpy` and `math` to overload elementory operations.

Our test suite will be in the `test` folder, and we will implement `pytest` to write comprehensive tests to provide full coverage for our code. We will also use `TravisCI` and `CodeCov` to automate the testing process.

PyPI will be used to distribute our package, as it enables the user to install our package using `pip`.

## Implementation

1. **What are the core data structures?**
The core data structures is an array of parameters with size of *K+1* where *K* is the number of input variables in the user-defined function. For example, we will store parameters *a* and *b_i (1<i<K)* for dual number ![dual](https://latex.codecogs.com/svg.latex?a+\sum_{i}^{K}{b_i\varepsilon_i}) if there are *K* input variables. 

1. **What classes will you implement?**
We will have two core classes as follows:
    - ***Var***: class that defines the dual number, and provides basic operators manipulating on dual numbers including overloaded build-in operators (eg, *, /, +, -, **) and more advanced functions (eg, sin, sqrt, log, exp). 
    - ***AD***: class that builds on top of ***Var*** and provides interface for users to calculate derivatives in different cases (eg, scalar functions of scalar values, vector functions of vector values, and scalar functions of vector values). 

1. **What method and name attributes will your classes have?**

    ***Var*** class will have an array of parameters which describe the dual number (ie, the derivatives against each input variable). It has a bunch of overloaded build-in functions (eg, *, /, +, -, **) and advanced functions (eg, sin, sqrt, log, exp) implementing dual number operations correspondingly. Specifically, it should have the following signature: 
    ```python
    class Var():
        def __init__(self, K, dual_paras):
            ...
        def __add__(self, other):
            ...
        def __radd__(self, other):
            ...
        ...
        def sin(self):
            ...
        def sqrt(self):
            ...
        ...                                
    ```
    where *K* is the number of input variables, and *dual_paras* describe one specific dual number. 

    ***AD*** class includes some functions that users can use to do AD-related calculation. For example, it can include automatic differentiation, Jacobian matrix calculation, etc. Specifically, it should have the following signature: 
    ```python
    class AD():
        def __init__(self):
            ...
        def auto_diff(self, func, Vars):
            ...
        def jac_matrix(self, func, Vars):
            ...        
        ...                                
    ```
    where *func* is a user-defined function (eg, `f1 = lambda x,y: x**2 + y`), and *Vars* is an array of ***Var*** instances (eg, instance x, y). 

1. **What external dependencies will you rely on?**
    We will rely on *numpy* and *math* library for mathematic operations, and *pytest* for testing purpose. 

1. **How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)?**
    As mentioned above, for python build-in operations, we overload them following the corresponding dual number operations; for other elementary functions, we implement them within **Var** class. 
