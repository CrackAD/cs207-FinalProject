import pytest
import numpy as np
from var import Var

def test_add_var_result():
	x = Var(3, np.array([1,0]))
	y = Var(2, np.array([0,1]))
	assert x + y == (5, np.array([1,1]))

def test_add_realn_result():
	x = 3
	y = Var(2, np.array([0,1]))
	assert x+y  == (5, np.array([0,1]))

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

test_add_var_result()
