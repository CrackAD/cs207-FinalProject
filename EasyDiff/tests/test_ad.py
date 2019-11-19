import pytest
import numpy as np
import var
from var import Var
from ad import AD

x = Var(3, np.array([1,0]))
y = Var(2, np.array([0,1]))

def test_single_variable():
	ad = AD(np.array([2]), np.array([1]))
	f1 = lambda x: Var.log(x) ** 2
	assert AD.auto_diff(self=ad,func=f1) == Var(pytest.approx(0.4804530139182014),np.array([pytest.approx(0.69314718)]))

def test_two_variables():
	ad = AD(np.array([2, 2]), np.array([1, 1]))
	f1 = lambda x, y: Var.log(x) ** Var.sin(y)
	assert AD.auto_diff(self=ad,func=f1) == Var(pytest.approx(0.7165772257590739),np.array([pytest.approx(0.47001694),pytest.approx(0.10929465)]))




