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
    assert x.grad() == pytest.approx(y.value + np.cos(x.value))
    assert y.grad() == pytest.approx(x.value)

def test_rev_add():
    x = Rev_Var(0.5)
    y = Rev_Var(4.2)
    assert x + y == Rev_Var(x.value+y.value)

def test_rev_real_add():
    x = Rev_Var(0.5)
    y = 4.2
    assert x + y == Rev_Var(x.value+4.2)

def test_rev_mul():
    x = Rev_Var(0.5)
    y = Rev_Var(4.2)
    z = x*y

    assert z == Rev_Var(0.5 * 4.2)


# def test_real_mul():
# 	x = 3
# 	y = Var(2, np.array([0,1]))
# 	assert x*y == Var(6, np.array([0,3]))

# def test_var_sub():
# 	x = Var(3, np.array([1,0]))
# 	y = Var(2, np.array([0,1]))
# 	assert x - y == Var(1, np.array([1,-1]))

# def test_real_sub():
# 	x = 3
# 	y = Var(2, np.array([0,1]))
# 	assert x-y == Var(1, np.array([0,-1]))

# def test_var_pow():
# 	x = Var(3, np.array([1,0]))
# 	y = Var(2, np.array([0,1]))
# 	assert x**y == Var(9, np.array([pytest.approx(6.0),pytest.approx(9.8875106)]))

# def test_real_pow():
# 	x = Var(3, np.array([1,0]))
# 	y = 2
# 	assert x**y == Var(9, np.array([6, 0]))
# 	assert y**x == Var(8, np.array([pytest.approx(5.54517744), pytest.approx(0.)]))

# def test_truediv():
# 	x = Var(3, np.array([1,0]))
# 	y = Var(2, np.array([0,1]))
# 	assert x/y == Var(1.5, np.array([0.5, -0.75]))

# def test_rtruediv():
# 	x = Var(3, np.array([1,0]))
# 	y = 2
# 	assert y/x == Var(pytest.approx(0.6666666666666666), np.array([pytest.approx(-0.22222222), pytest.approx(-0.)]))

# def test_div():
# 	x = Var(3, np.array([1,0]))
# 	y = 2
# 	assert x/y == Var(1.5, np.array([0.5, 0.]))

# def test_neg():
# 	x = Var(3, np.array([1,0]))
# 	assert -x == Var(-3, np.array([-1,0]))

# def test_pos():
# 	x = Var(3, np.array([1,0]))
# 	assert +x == Var(3, np.array([1,0]))

# def test_equal():
# 	x = Var(3, np.array([1,0]))
# 	z = Var(3, np.array([1,0]))
# 	y = 2
# 	assert x==z
# 	assert (x==y) == False

# def test_notequal():
# 	x = Var(3, np.array([1,0]))
# 	y = 2
# 	z = Var(2, np.array([0,1]))
# 	assert x!=z
# 	assert (x!=y) == True

# def test_log():
# 	x = Var(3, np.array([1]))
# 	y = 2
# 	assert Var.log(x) == Var(pytest.approx(1.0986122886681098), np.array([pytest.approx(0.33333333)]))
# 	assert Var.log(y) == np.log(y)

# def test_logk():
# 	x = Var(3, np.array([1]))
# 	y = 2
# 	assert Var.logk(x,3.0) == Var(pytest.approx(1.0), np.array([pytest.approx(0.30341308)]))
# 	assert Var.logk(y,3.0) == np.log(y)/ np.log(3.0)

# def test_exp():
# 	x = Var(3, np.array([1]))
# 	y = 2
# 	assert Var.exp(x) == Var(pytest.approx(20.085536923187668), np.array([pytest.approx(20.08553692)]))
# 	assert Var.exp(y) == np.exp(y)

# def test_sqrt():
# 	x = Var(3, np.array([1]))
# 	y = 2
# 	assert Var.sqrt(x) == Var(pytest.approx(1.7320508075688772), np.array([pytest.approx(0.28867513)]))
# 	assert Var.sqrt(y) == np.sqrt(y)

# def test_sin():
# 	x = Var(3, np.array([1]))
# 	y = 2
# 	assert Var.sin(x) == Var(pytest.approx(0.1411200080598672), np.array([pytest.approx(-0.9899925)]))
# 	assert Var.sin(y) == np.sin(y)

# def test_cos():
# 	x = Var(3, np.array([1]))
# 	y = 2
# 	assert Var.cos(x) == Var(pytest.approx(-0.9899924966004454), np.array([pytest.approx(-0.14112001)]))
# 	assert Var.cos(y) == np.cos(y)

# def test_tan():
# 	x = Var(3, np.array([1]))
# 	y = 2
# 	assert Var.tan(x) == Var(pytest.approx(-0.1425465430742778), np.array([pytest.approx(1.02031952)]))
# 	assert Var.tan(y) == np.tan(y)


