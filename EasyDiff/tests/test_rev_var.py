import pytest
import numpy as np
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from rev_var import Rev_Var


def test_rev_grad():
    x = Rev_Var(0.5)
    y = Rev_Var(4.2)
    z = x * y + Rev_Var.sin(x)
    z.grad_value = 1.0

    assert z.value == pytest.approx(2.579425538604203)
    assert x.grad() == pytest.approx(4.2 + np.cos(x.value))
    assert y.grad() == pytest.approx(0.5)

def test_rev_add():
    x = Rev_Var(0.5)
    y = Rev_Var(4.2)
    z = x+y
    z.grad_value = 1.0
    assert z.value == 4.7
    assert y.grad() == 1

def test_rev_real_add():
    x = Rev_Var(0.5)
    y = 4.2
    z = x+y
    assert z.value == x.value + y

def test_rev_mul():
    x = Rev_Var(0.5)
    y = Rev_Var(4.2)
    z = x*y
    z.grad_value = 1.0
    assert z.value == 2.1
    assert x.grad() == pytest.approx(y.value)

def test_rev_rmul():
	x = 3
	y = Rev_Var(4.2)
	z = x*y
	assert z.value == pytest.approx(12.6)

def test_rev_sub():
    x = Rev_Var(0.5)
    y = Rev_Var(4.2)
    a = 3

    z1 = y - x
    z1.grad_value = 1.0
    assert z1.value == 3.7
    assert x.grad() == -1
    assert y.grad() == 1

    z2 = y - 3
    assert z2.value == pytest.approx(1.2)

def test_rev_rsub():
    x = Rev_Var(0.5)
    a = 3
    z = a - x
    assert z.value == pytest.approx(2.5)

def test_rev_pow():
    x1 = Rev_Var(0.5)
    x2 = Rev_Var(0.5)
    y = Rev_Var(4.2)
    a = 3
    z1 = x1 ** y
    z1.grad_value = 1
    z2 = x2**3
    z2.grad_value = 1

    assert z1.value == pytest.approx(0.05440941020600775)
    assert x1.grad() == pytest.approx(0.4570390457304651)
    assert y.grad() == pytest.approx(-0.03771372928022378)
    assert z2.value == pytest.approx(0.125)
    assert x2.grad() == pytest.approx(0.75)


def test_rev_rpow():
    x = Rev_Var(0.5)
    a = 3
    z = 3 ** x
    z.grad_value = 1
    assert z.value == pytest.approx(1.7320508075688772)
    assert x.grad() == pytest.approx(1.902852301792692)

def test_truediv():
    x = Rev_Var(0.5)
    y = Rev_Var(4.2)
    z = x/y
    z.grad_value = 1.0
    assert z.value == pytest.approx(0.11904761904761904)
    assert x.grad() == pytest.approx(0.23809523809523808)
    assert y.grad() == pytest.approx(-0.028344671201814057)

def test_rtruediv():
	x = Rev_Var(0.5)
	y = 5
	z = x/y
	z.grad_value = 1.0
	assert z.value == 0.1
	assert x.grad() == 0.2

def test_neg():
	x = Rev_Var(0.5)
	y = Rev_Var(-0.5)
	assert -x.value == -0.5
	assert x.value == -y.value


def test_pos(): 
	x = Rev_Var(0.5)
	assert +x.value == x.value
	assert +x.value == 0.5

def test_equal():
	x = Rev_Var(0.5)
	z = Rev_Var(0.5)
	y = 2
	assert x==z
	assert (x==y) == False

def test_notequal():
	x = Rev_Var(0.5)
	y = 2
	z = Rev_Var(4)
	assert x!=z
	assert (x!=y) == True

def test_log():
	x = Rev_Var(2.0)
	z1 = Rev_Var.log(x)
	z1.grad_value = 1
	z2 = Rev_Var.log(2.0)
	assert z1.value == pytest.approx(0.6931471805599453)
	assert z1.value == np.log(2)
	assert x.grad() == pytest.approx(0.5)
	assert z2 == np.log(2)

def test_logk():
	x = Rev_Var(2.0)
	z2 = Rev_Var.logk(x, 3.0)
	z2.grad_value = 1
	z3 = Rev_Var.logk(2.0, 3.0)
	assert z2.value == np.log(2) / np.log(3)
	assert x.grad() == pytest.approx(0.45511961331341866)
	assert z3 == np.log(2)/np.log(3)

def test_exp():
	x = Rev_Var(2.0)
	z1 = Rev_Var.exp(x)
	y = 5
	z1.grad_value = 1
	z2 = Rev_Var.exp(y)
	assert z1.value == pytest.approx(7.38905609893065)
	assert z2 == pytest.approx(148.4131591025766)
	assert x.grad() == pytest.approx(7.38905609893065)

def test_expk():
	x = Rev_Var(2.0)
	z = Rev_Var.expk(3.0,x)
	z.grad_value = 1
	assert z.value == pytest.approx(8)
	assert x.grad() == pytest.approx(12)

def test_sqrt():
	x = Rev_Var(2.0)
	z1 = Rev_Var.sqrt(x)
	z1.grad_value = 1
	z2 = Rev_Var.sqrt(2)
	assert z1.value == pytest.approx(1.4142135623730951)
	assert x.grad() == pytest.approx(0.3535533905932738)
	assert z2 == np.sqrt(2)

def test_logistic():
	x = Rev_Var(3.0)
	z = Rev_Var.logistic(x)
	z.grad_value = 1
	z2 = Rev_Var.logistic(3.0)
	assert z.value == pytest.approx(1 / (1+np.exp(-3)))
	assert x.grad() == pytest.approx(np.exp(3) / ((1 + np.exp(3))**2))
	assert z2 == 1 / (1 + np.exp(-3.0))

def test_sinh():
	x = Rev_Var(2.0)
	z = Rev_Var.sinh(x)
	z.grad_value = 1
	z2 = Rev_Var.sinh(2)
	assert z.value == pytest.approx(3.626860407847019)
	assert z2 == pytest.approx(3.626860407847019)
	assert x.grad() == pytest.approx(3.7621956910836314)

def test_cosh():
	x = Rev_Var(2.0)
	z = Rev_Var.cosh(x)
	z.grad_value = 1
	z3 = Rev_Var.cosh(2)
	assert z.value == pytest.approx(3.7621956910836314)
	assert z3 == pytest.approx(3.7621956910836314)
	assert x.grad() == pytest.approx(3.626860407847019)

def test_tahh():
	x = Rev_Var(2.0)
	z = Rev_Var.tanh(x)
	z.grad_value = 1
	z4 = Rev_Var.tanh(2)
	assert z.value == pytest.approx(0.964027580075817)
	assert z4 == (np.exp(2) - np.exp(-2)) / (np.exp(2) + np.exp(-2))
	assert x.grad() == pytest.approx(0.07065082485316432)

def test_sin():
	x = Rev_Var(2.0)
	y = 2
	z = Rev_Var.sin(x)
	z.grad_value = 1
	assert z.value == pytest.approx(0.9092974268256817)
	assert x.grad() == pytest.approx(-0.4161468365471424)
	assert Rev_Var.sin(y) == np.sin(y)

def test_cos():
	x = Rev_Var(2.0)
	y = 2
	z = Rev_Var.cos(x)
	z.grad_value = 1
	assert z.value == pytest.approx(-0.4161468365471424)
	assert x.grad() == pytest.approx(-0.9092974268256817)
	assert Rev_Var.cos(y) == np.cos(y)

def test_tan():
	x = Rev_Var(2.0)
	y = 2
	z = Rev_Var.tan(x)
	z.grad_value = 1
	assert z.value == pytest.approx(-2.185039863261519)
	assert x.grad() == pytest.approx(5.774399204041917)
	assert Rev_Var.tan(y) == np.tan(y)

def test_arcsin():
	x = Rev_Var(0.5)
	y = 0.3
	z = Rev_Var.arcsin(x)
	z.grad_value = 1
	assert z.value == pytest.approx(0.5235987755982988)
	assert x.grad() == pytest.approx(1.1547005383792517)
	assert Rev_Var.arcsin(y) == np.arcsin(y)

def test_arccos():
	x = Rev_Var(0.5)
	y = 0.3
	z = Rev_Var.arccos(x)
	z.grad_value = 1
	assert z.value == pytest.approx(1.0471975511965976)
	assert x.grad() == pytest.approx(-1.1547005383792517)
	assert Rev_Var.arccos(y) == np.arccos(y)

def test_arctan():
	x = Rev_Var(0.5)
	y = 0.25
	z = Rev_Var.arctan(x)
	z.grad_value = 1
	assert z.value == pytest.approx(0.46364760900080615)
	assert x.grad() == pytest.approx(0.8)
	assert Rev_Var.arctan(y) == np.arctan(y)



