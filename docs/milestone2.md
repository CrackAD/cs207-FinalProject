## Introduction

Across a wide range of applications, taking derivatives is essential. For example, we take derivatives on a regular basis in optimization problems where we need to find maxima and minima. In computer programming, there are four main ways to compute derivatives: 

- Manually evaluating them and coding them
- Performing numerical differentiation using finite difference approximations
- Using symbolic differentiation using expression manipulation
- Using automatic differentiation
 
The first three methods have key shortcomings. Manual differentiation is time consuming. Numerical differentiation can be simple to implement but can be highly inaccurate due to rounding errors. Finally, symbolic differentiation may lead to overly complicated expressions.
 
In contrast, automatic differentiation has several advantages. It doesn’t require us to run a single expression; we can break things up into manageable parts using the chain rule. It avoids round-off errors that can be introduced by numerical differentiation, leading to more accurate outputs. It can also handle higher derivatives more easily and do so more efficiently; the rules of differentiation and evaluation can be carried out in parallel, requiring only small amounts of extra storage.
 
In this project, we aim to create an automatic differentiation package so that we can not only run automatic differentiation but also make the software available to others at scale by way of an easy-to-implement python package.

## Background

#### Automatic Differentiation & The Forward Mode
Given the notion that AD enables us to compute the derivatives of a function efficiently and accurately, let us dig further into its mathematical mechanism:

**The Chain Rule**

The chain rule is fundamental to AD. It decomposes the derivative calculation for complex functions with multiple layers. A simple example goes as follows:

```
Let F(x) = f(g(x))
```
The derivative of F(x) is, by the chain rule:
```
F'(x)=f'(g(x))g'(x)
```
This expression can be easily expanded to when we need to apply the chain rule to composites of more than two functions, facilitating the computation of derivatives.

**Elementary Function**

An elementary function is a simple function of one variable. 

Examples include:
* arithmetic operations (+ – × /)
* exponentials, logarithms, triangle functiosns, etc
* constants

**Forward Mode**

Forward mode, as the name suggests, traverses the chain rule from the inside to the outside of the function. For each step, it calculates a function's current value, as well as the numeric value of this step's elementary function's derivative. In other words, the derivatives are computed in sync with the evaluation steps and are combined with other derivatives via the chain rule. Therefore, the forward mode is easy to understand and implement. (Note that forward mode is less efficient with a large number of parameters.)

#### Dual Number & AD

The application of dual numbers (![dual](https://latex.codecogs.com/svg.latex?a+b\epsilon)) is a neat trick in computing AD. We can use dual number operations on numbers to calculate the value of f(x) while calculating f'(x) at the same time. In this way, we get the derivative directly, and the value of the function at the same time, without having to go through the forward mode step by step.

The key to the method is converting x into a dual number, using 1 for the dual component, since we are plugging it in for the value of x, which has a derivative of 1.

In this way, the final solution has the evaluation result (the real component), as well as the derivative in terms of x (the dual component).

#### Multivariate Dual Number & AD

It's also intuitive to use dual numbers with multivariable functions. Since the expected end result is a partial derivative for each variable in the equation, we would just compute a dual number per variable, and process the entire equation for each of those dual numbers separately.

For instance, let's say we want to calculate the partial derivatives of x and y of the function ![dual](https://latex.codecogs.com/svg.latex?3x^2-2y^3) with the input (5, 2). First, to get the partial derivative of x, we substitue x with ![dual](https://latex.codecogs.com/svg.latex?5+1\epsilon_x+0\epsilon_y) and y with ![dual](https://latex.codecogs.com/svg.latex?2+0\epsilon_x+1\epsilon_y) (when calculating the partial derivative of x, y is a constant). This gives us ![dual](https://latex.codecogs.com/svg.latex?59+30\epsilon_x-24\epsilon_y), saying that the value is 59 at location (5, 2), and the derivative of x at that point is 30 and the partial derivative of y to be -24.


## How to Use EasyDiff
First the user needs to install the package using the following command:

```bash
pip install EasyDiff
```

After installing the package, the user should import it to perform the forward mode calculations:

```python
from EasyDiff import Var
from EasyDiff import AD
```

There are two main classes in our EasyDiff package: ***Var*** and ***AD***. The user should import both classes. The user should then use ***Var*** to create input variables, and use ***AD*** to calculate the derivatives, Jacobian matrix, etc. An example of using EasyDiff is shown as follows: 

```python
from EasyDiff import Var
from EasyDiff import AD

K = 2 # two input variables
f1 = lambda x,y: x**2 + y
x = Var(K, [2, 1]) # evaluating at x=2 with initial derivative of 1
y = Var(K, [5, 1]) # evaluating at y=5 with initial derivative of 1

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

* `ad.py`: interfaces for automatic-differentiation-related calculations (eg, Jacobian matrix)
* `var.py`: dual number constructions, basic math operations overloaded, and other elementary functions. 

We also plan on including dependencies `numpy` and `math` to overload elementory operations.

Our test suite will be in the `test` folder, and we will implement `pytest` to write comprehensive tests to provide full coverage for our code. We will also use `TravisCI` and `CodeCov` to automate the testing process.

PyPI will be used to distribute our package, as it enables the user to install our package using `pip`.

## Implementation

#### Core Classes, Important Attributes, and Data Structures

We have two classes as follows: 
- ***Var***: class that contains the value and dual numbers, and provides basic operators manipulating on dual numbers including overloaded build-in operators (eg, *, /, +, -, **) and other elementary functions (eg, sin, sqrt, log, exp). 
- ***AD***: class that builds on top of ***Var*** and provides interface for users to calculate derivatives in different cases (eg, scalar functions of scalar values, vector functions of vector values, and scalar functions of vector values). 

***Var*** has two attributes: the value and the dual numbers (ie, the derivatives against each input variable). The value is a single scale, while the dual numbers are an numpy array of dual numbers with size of *K* where *K* is the number of input variables in the user-defined function. 
For example, we will store parameters *a* and *b_i (1<i<K)* for dual number ![dual](https://latex.codecogs.com/svg.latex?a+\sum_{i}^{K}{b_i\varepsilon_i}) if there are *K* input variables. 

***AD*** has a attributes: the Var object array. 


***Var*** class has a bunch of overloaded build-in functions (eg, *, /, +, -, **) and other elementary functions (eg, sin, sqrt, log, exp) implementing dual number operations correspondingly. We implement these elementary functions as ***static methods***. 
Specifically, it should have the following signature: 
```python
class Var():
    def __init__(self, val, dual_paras):
        ...
    def __add__(self, other):
        ...
    def __radd__(self, other):
        ...
    ...
    @staticmethod
    def sin(self):
        ...
    @staticmethod
    def sqrt(self):
        ...
    ...                                
```

***AD*** class includes some functions that users can use to do AD-related calculation. For example, it can include automatic differentiation, Jacobian matrix calculation, etc. Specifically, it should have the following signature: 
```python
class AD():
    def __init__(self, vals, ders):
        ...
    def auto_diff(self, func):
        ...
    def jac_matrix(self, funcs):
        ...        
    ...                                
```
where *vals* is an array of the initial values for the *K* input variables, *ders* is an array of the initial derivatives for the *K* input variables, *func* and *funcs* are the input functions (eg, `f1 = lambda x,y: x**2 + y`). 

#### External dependencies and Elementary functions

We will rely on *numpy* library for mathematic operations, and *pytest*, *pytest-cov*, and *doctest* for testing purpose. 

Currently, we cover the following elementary functions: 
* **\+** (\_\_add\_\_, \_\_radd\_\_), **-** (\_\_sub\_\_, \_\_rsub\_\_), **\*** (\_\_mul\_\_, \_\_rmul\_\_), **/** (\_\_truediv\_\_, \_\_rtruediv\_\_), **\*\*** (\_\_pow\_\_, \_\_rpow\_\_), 
* \_\_neg\_\_, \_\_pos\_\_, \_\_eq\_\_, \_\_ne\_\_, 
* log(), logk, sqrt(), exp(),
* sin(), cos(), tan().

For python build-in operations (the first two rows), we overload them following the corresponding dual number operations; 
for other elementary functions (the last two rows), we implement them within **Var** class as static methods. 
