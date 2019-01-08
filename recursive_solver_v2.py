

import numpy as np
from numpy.polynomial.polynomial import polyval
from numpy.linalg import matrix_power
from numpy.linalg import matrix_rank
from scipy.linalg import lu
from scipy.linalg import pascal
import sympy

def solve_poly(A, q, x0, y0):

	# takes upper triangular A

	# remove uneccisary columns
	A,q = A[:-1,1:], q[:-1]

	# solve system for coeffients a1->an
	p = np.zeros(shape=(len(q)+1,))
	p[1:] = np.linalg.solve(A,q)

	# solve system for final coefficient
	y_pred = polyval(x0, p, tensor=True)

	# find a0
	p[0] = y0 - y_pred

	return p


# recursion: p(x +h) = p(x) + q(x+h), p(x0) = y0
def interpolate_recursion(q, h, x0, y0):
	# q is list of coefficients on [x^0, x^1, ..., x^{n-1}]

	# get degree of p, n, from degree of q, n-1
	n = len(q)

	# get shift matrix
	D_h = basic_D(n+1, h) 

	# turn q(x) into q(x+h) and add row to q with zero element
	q_h = np.zeros(shape=(n+1,))
	q_h[:-1] = q
	#q_h = D_h @ q_h


	# Find A = D_h - I, from x(D_h - I)p = x(D_h)q
	I = np.eye(n+1)
	A = D_h - I

	# solve for p
	p = solve_poly(A, q_h, x0, y0)

	#return p
	return np.poly1d(p[::-1])


def power_sum( n, d=1 ):
	""" power_sum( n ) - power_sum( n-1 ) = n**d """

	if np.isscalar(n):
		# must be an integer
		n = int(n)
		s = 0
		for i in range(1,n+1): 
			s += i**d
		return s

	return np.array([ q( ni, d=d ) for ni in n ])

def power_sum_polynomial(degree):

	# step size
	h = 1

	# any initial x value
	x0 = 4

	# known y
	y0 = power_sum(x0, d=degree)

	# create array of coefficients for p(x+h) - p(x) = (x+h)^degree
	q = np.zeros(shape=(degree+1,))
	q[-1] = 1
	D_h = basic_D(degree+1, 1)
	q = D_h @ q

	# solve for polynomial
	p = interpolate_recursion(q, h, x0, y0)

	return p


# define a translation matrix Lh of size nxn
def poly_translation_matrix(n, h, kind='lower'):

	# pascal matrix
	P = pascal(n, kind='lower')

	# exponent matrix
	L = np.tril(np.ones(shape=(n,n)))
	R = np.tril(matrix_power(L, 2)-1)

	# H matrix, H's raised to corresponding powers
	H = np.tril(np.power(h,R)) 

	# Create S poly shift matrix with offset h
	Lh = (H * P) 

	if kind == 'lower':
		return Lh

	# return upper triangular
	Uh = Lh.T
	return Uh

from functools import reduce
# U_1 + U_2 + ... + U_i
def stacked_translation_matrix(n, h, n_terms, kind='lower'):

	# q(x) = p(x + h) + p(x + 2h) + ... + p(x + ih) -- i == steps
	Us = [poly_translation_matrix(n,k, kind='upper') for k in range(1,n_terms+1)] 

	# U_1 + U_2 + ... + U_i
	U = reduce(lambda a,b : a+b,Us)

	return U

def translate_polynomial(p, h):

	# q(x) = p(x + h)
	Lh = poly_translation_matrix(len(p)+1,h)
	coefs = Lh.T @ p.c[::-1]
	q = np.poly1d( coefs[::-1] )

	return q



from functools import reduce
def sucsessive_polynomial_combination(p, h, steps, output_degree):

	# q(x) = p(x + h) + p(x + 2h) + ... + p(x + ih) -- i == steps
	Us = [poly_translation_matrix(p_degree,k, kind='upper') for k in range(1,steps+1)] 

	# U_1 + U_2 + ... + U_i
	U = reduce(lambda a,b : a+b,Us)

	return U @ p


# settings
q_degree = 2
p_degree = q_degree + 1
h = 1
x0 = 40
y0 = power_sum(x0, d=q_degree)

I = np.eye(p_degree)
Lh = poly_translation_matrix(p_degree,h)
Lh_inv = np.linalg.inv(Lh)
A = (Lh-I)@Lh_inv

q = np.zeros(shape=(p_degree,))
q[-2] = 1
q_h = Lh.T @ q

p = solve_poly(A.T, q_h, x0, y0)
p = np.poly1d(p[::-1])

print(p)
print(p(x0) - y0)
