import unittest
import numpy.testing as npt 

from mumfordshah import *

class TestMumford(unittest.TestCase):

	#def setup(self):
	#	nc = 5
	#	#Params from [1]
	#	theta = 1
	#	tau = 0.01
	#	h = 1
	#	L2 = 8/h**2
	#	sigma = 1/(L2 * tau)
	#	lmda = 5e-6						#Not sure what this should be set to....
	#	ny = 512

	def test_grad(self):
		a = np.ones((4,4,3))
		h = 2
		a[0:2,0:2,:] = 0
		result = grad(a, h)
		answer = np.zeros((4,4,2,3))
		answer[1,0,0,:] = 0.5
		answer[1,1,0,:] = 0.5
		answer[0,1,1,:] = 0.5
		answer[1,1,1,:] = 0.5
		npt.assert_almost_equal(answer, result)

	def test_div(self):
		a = np.zeros((4,4,2,3))
		a[1,0,0,:] = 0.5
		a[1,1,0,:] = 0.5
		a[0,1,1,:] = 0.5
		a[1,1,1,:] = 0.5
		h = 2
		result = div(a, h)
		answer = np.zeros((4,4,3))
		answer[0,0,:] = 0.5
		answer[1,1,:] = -0.5
		npt.assert_almost_equal(answer, result)

	def test_project_simplex(self):
		a = np.zeros((2,1,4))
		b = np.zeros((2,1,3))
		c = np.zeros((2,1,1))
		ans_a = np.zeros((2,1,4))
		ans_b = np.zeros((2,1,3))
		ans_c = np.zeros((2,1,1))
		a[0,0,:] = [1, 1, 1, 1]
		a[1,0,:] = [2, 2, 1, 1]
		b[0,0,:] = [0, 0, 0]
		b[1,0,:] = [1, 0, 1]
		c[0,0,:] = [1]
		c[1,0,:] = [0]
		ans_a[0,0,:] = [1/4., 1/4., 1/4., 1/4.]
		ans_a[1,0,:] = [1/2., 1/2., 0, 0]
		ans_b[0,0,:] = [1/3., 1/3., 1/3.]
		ans_b[1,0,:] = [1/2., 0, 1/2.]
		ans_c[0,0,:] = [1]
		ans_c[1,0,:] = [1]
		npt.assert_almost_equal(project_simplex(a), ans_a)
		npt.assert_almost_equal(project_simplex(b), ans_b)
		npt.assert_almost_equal(project_simplex(c), ans_c)

	def test_project_dykstra(self):
		assert True 