## Background

###Automatic Differentiation & The Forward Mode
So we know that AD enables us to compute the derivatives of a function efficiently and accurately. Let's now dig further into its mathematical mechanism:

**The Chain Rule**

The chain rule serves as the Fundamentals of AD. It decomposes the derivative calculation for complex functions with multiple layers. A simple example goes as follows:

```{bash}
Let F(x) = f(g(x))
The the derivative of F(x) is, by the chain rule
F'(x)=f'(g(x))g'(x)
```
This can be easily expanded to apply to composites of more than two functions and largely easy the computation of derivatives.

**Elementary Function**

An elementary function is a simple function of one variable. 

Examples include:
* arithmetic operations (+ – × ÷)
* exponentials, logarithms, roots
* constants

**The Forward Mode**

The forward mode, as the name suggests, traverses the chain rule from inside to outside. For each step, it calculates a function's current value, as well as the numeric value of this step's elementary function's derivative. In another word, the derivatives are computed in sync with the evaluation steps and combined with other derivatives via the chain rule. Therefore, the forward mode is easy to understand and implement. (note that it is less efficient with a large number of parameters)

###Dual Number & AD

The application of dual numbers (a+εb) is a neat trick in computing AD. We can use dual number operations on numbers to calculate the value of f(x) while calculating f'(x) at the same time. 

The key to the method is converting x into a dual number, using 1 for the dual component, since we are plugging it in for the value of x, which has a derivative of 1.

In this way, the final solution has the evaluation result (the real component), as well as the derivative in terms of x (the dual component).

###Multivariate Dual Number & AD

It's also intuitive to use dual numbers with multivariable functions. Since the expected end result is a partial derivative for each variable in the equation, we would just compute a dual number per variable, and process the entire equation for each of those dual numbers separately.

For instance, let's say we want to calculate the partial derivatives of x and y of the function 3x^2-2y^3 with the input (5,2). First, to get the partial derivative of x, we substitue x with 5+1ε and y with 2+0ε (when calculating the partial derivative of x, y is a constant). This gives us 59+30ε, saying that the value is 59 at location (5,2), and the derivative of x at that point is 30.

In the same way, we compute the partial derivative of y to be -24 at (5,2)
```