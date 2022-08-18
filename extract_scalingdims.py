import numpy as np
import scipy.optimize, scipy.stats
from time import perf_counter
import csv
import os
import glob


pi = np.pi

def hankel_matrix(y):
	"""Generate a Hankel matrix given a sequence of numbers"""
	L = int(np.floor(len(y) / 2))
	M = len(y) - 1

	H = np.zeros((L+1, M-L+1))

	left, right = 0, M-L+1
	for i, row in enumerate(H):
		H[i] = y[left:right]
		left += 1
		right += 1
	return H

def ESPRIT(y, q):
	"""
	ESPRIT algorithm (substep is auxiliary):
	1. Compute Hankel matrix of y: H(y)
	2. Perform SVD on H(y) to get [U U_orth], s, [V V_orth]*
		2.1. Obtain W by taking the first S columns of [U U_orth]
	3. Get U_0 by taking first L rows from U, get U_1 by taking last L rows from U
	4. Psi = np.linalg.pinv(W_0) @ W_1
	5. Compute eigenvalues of Psi

	From https://arxiv.org/pdf/1905.03782.pdf
	"""

	H = hankel_matrix(y)

	U, s, V = np.linalg.svd(H)

	L = int(np.floor(len(y) / 2))
	W = U[:,:q]
	W_0 = W[:L] # first L rows of W
	W_1 = W[-L:] # last L rows of W

	Psi = np.linalg.pinv(W_0) @ W_1
	eigv = np.linalg.eig(Psi)[0]
	return eigv

def sum_of_evs(x, *factors):
	"""
	Performs sum over eigenvalues to the power of k
	Used as fit function to optimize the factors
	"""

	y = np.array([])
	for k in x:
		y = np.append(y, np.sum(factors[:len(ev)]*(ev**k)))
	return y / y[0] # normalize

def get_RSS(E, q):
	"""Find residual sum-of-squares for model with {order} eigenvalues"""

	global ev
	ev = np.real(ESPRIT(E, q))

	# popt are the optimized factors of sum_of_evs() fitted to the observable decay curve
	# p0 = [1]*order to fit exactly {order} eigenvalues	
	popt = scipy.optimize.curve_fit(sum_of_evs, range(len(E)), E, p0 = [1]*q)[0]
	RSS = np.sum((E - sum_of_evs(range(len(E)), *popt))**2)
	return RSS

def F_test(E, q):
	"""
	Compute the F statistic and perform the F test to 
	determine whether to add more eigenvalues to the fit
	"""

	n = len(E)
	RSS1 = get_RSS(E, q)
	RSS2 = get_RSS(E, q+1)
	
	# Compute F statistic
	dfn = 1 	# degrees of freedom numerator
	dfd = n - (q+1)		# degrees of freedom denominator
	F = (RSS1 - RSS2)/dfn / (RSS2/dfd)

	# If obtained F statistic is greater than critical F value, 
	# then model with more parameters is statistically more accurate
	if F > scipy.stats.f.ppf(q=1-0.05, dfn=dfn, dfd=dfd):
		return True
	else:
		return False

def comp_evals_from_obs_curve(E):
	"""
	Obtain eigenvalues from observable decay curve with ESPRIT,
	then remove non-relevant eigenvalues with F test
	"""

	# Keep adding eigenvalues until it does not statistically improve fit
	t = 1
	while True:
		if F_test(E, t):
			t += 1
			continue
		else:
			return np.real(ESPRIT(E, t))


def comp_evals_from_datafile(filename):
	"""
	Given a file with many decay curves for an observable,
	compute the eigenvalues for all starting states
	"""

	all_evals = []

	if os.path.isfile(filename):
		with open(filename, newline="") as f:
			cr = csv.reader(f)
			for i, row in enumerate(cr):
				if i == 0:
					max_len = len(row)+3
					total_E = np.zeros(max_len)
				E = [float(val) for val in row[4:]]

				total_E += np.pad(E, (0,max_len-len(E)), constant_values=E[-1])

			total_E = total_E[:-5]
			evals = comp_evals_from_obs_curve(total_E)
			all_evals += list(evals)
			print(f"Evals = {evals}")
	else:
		print("File does not exist.")

	return all_evals


def comp_evals_from_all_files(decomp="exact", decomp_val=0):
	""""
	Combine all files for a given decomposition (and decomposition value)
	and return all eigenvalues.
	"""

	evals = []

	# list of all files with given decomposition
	list_files = glob.glob(f"./observable_data/*_data_{decomp}*")

	for filename in list_files:
		print(filename)
		evals += comp_evals_from_datafile(filename)

	return evals






