import pytest
import numpy as np
from EasyDiff.var import Var
from EasyDiff.ad import AD

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

def test_jac_matrix():
	f1 = lambda x, y: Var.log(x) ** Var.sin(y)
	f2 = lambda x, y: Var.sqrt(x) / y
	ad = AD(np.array([4.12, 5.13]), np.array([1, 1]))
	assert np.array_equal(ad.jac_matrix([f1, f2]), np.array([[pytest.approx(-0.11403015), pytest.approx(0.10263124)], [pytest.approx(0.048018), pytest.approx(-0.07712832)]]))
        
