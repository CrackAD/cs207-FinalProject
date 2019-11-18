import pytest
import numpy as np
import var
from var import Var


def test_var_add():
	x = Var(3, np.array([1,0]))
	y = Var(2, np.array([0,1]))
	assert x + y == Var(5, np.array([1,1]))

def test_real_add():
	x = 3
	y = Var(2, np.array([0,1]))
	assert x+y == Var(5, np.array([0,1]))

def test_var_mul():
	x = Var(3, np.array([1,0]))
	y = Var(2, np.array([0,1]))
	assert x*y == Var(6, np.array([2,3]))

def test_real_mul():
	x = 3
	y = Var(2, np.array([0,1]))
	assert x*y == Var(6, np.array([0,3]))

def test_var_sub():
	x = Var(3, np.array([1,0]))
	y = Var(2, np.array([0,1]))
	assert x - y == Var(1, np.array([1,-1]))

def test_real_sub():
	x = 3
	y = Var(2, np.array([0,1]))
	assert x-y == Var(1, np.array([0,-1]))

def test_var_pow():
	x = Var(3, np.array([1,0]))
	y = Var(2, np.array([0,1]))
	assert x**y == Var(9, np.array([pytest.approx(6.0),pytest.approx(9.8875106)]))

def test_real_pow():
	x = Var(3, np.array([1,0]))
	y = 2
	assert x**y == Var(9, np.array([6, 0]))

def test_truediv():
	x = Var(3, np.array([1,0]))
	y = Var(2, np.array([0,1]))
	assert x/y == Var(1.5, np.array([0.5, -0.75]))

def test_rtruediv():
	x = Var(3, np.array([1,0]))
	y = 2
	assert y/x == Var(pytest.approx(0.6666666666666666), np.array([pytest.approx(-0.22222222), pytest.approx(-0.)]))

def test_neg():
	x = Var(3, np.array([1,0]))
	assert -x == Var(-3, np.array([-1,0]))

def test_pos():
	x = Var(3, np.array([1,0]))
	assert +x == Var(3, np.array([1,0]))

def test_equal():
	x = Var(3, np.array([1,0]))
	z = Var(3, np.array([1,0]))
	y = 2
	assert x==z
	assert (x==y) == False

def test_notequal():
	x = Var(3, np.array([1,0]))
	y = 2
	z = Var(2, np.array([0,1]))
	assert x!=z
	assert (x!=y) == True

def test_log():
	x = Var(3, np.array([1,0]))
	assert Var.log(x) == Var(pytest.approx(1.0986122886681098), np.array([0.33333333])) 




# def test_quadroots_types():
#     with pytest.raises(TypeError):
#         roots.quad_roots("", "green", "hi")

# def test_quadroots_zerocoeff():
#     with pytest.raises(ValueError):
#         roots.quad_roots(a=0.0)

# def test_linearoots_result():
#     assert roots.linear_roots(2.0, -3.0) == 1.5

# def test_linearroots_types():
#     with pytest.raises(TypeError):
#         roots.linear_roots("ocean", 6.0)

# def test_linearroots_zerocoeff():
#     with pytest.raises(ValueError):
#         roots.linear_roots(a=0.0)
