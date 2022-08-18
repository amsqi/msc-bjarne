import numpy as np
import scipy.linalg
import string

pi = np.pi

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = 1j * X @ Z

def kron(*args):
    assert len(args) >= 1
    X = args[0]
    for Y in args[1:]:
        X = np.kron(X, Y)
    return X

# performs partial trace on i-th qubit of n-qubit state and returns reduced state
def part_trace(rho, i):
    n_ind = int(np.log2(rho.size)) # number of indices
    n_bits = int(n_ind / 2) # number of qubits

    # bring state to form "abcAbC" where "b" is the qubit-to-be-traced-over 
    rho = rho.reshape(np.full(n_ind, 2))
    dim_a = int(np.prod(rho.shape[:i]))
    dim_c = int(np.prod(rho.shape[i+1:n_bits]))
    rho = np.einsum("abcAbC->acAC", rho.reshape(dim_a, 2, dim_c, dim_a, 2, dim_c))

    return rho.reshape(2**(n_bits-1), 2**(n_bits-1))

# Molmer-Sorenson gate
def MS_XX(theta):
    return scipy.linalg.expm(-1j*theta/2*kron(X,X))

# single qubit Z rotation
def Z_rot(theta):
    return scipy.linalg.expm(1j*theta/2*Z)

# depolarizing channel on i-th qubit
def depol(rho, i, p):
    n_ind = int(np.log2(rho.size)) # number of indices
    n_bits = int(n_ind / 2) # number of qubits
    ind = string.ascii_lowercase[:n_ind-2] + ",xy"

    # replace the i-th qubit with maximally mixed state 
    depol_part = np.einsum(ind+"->"+ind[:i]+ind[-2]+ind[i:i-1+n_bits]+ind[-1]+ind[i-1+n_bits:-3], \
                 part_trace(rho, i).reshape(np.full(n_ind-2, 2)), I/2)

    return (1 - 4*p/3) * rho + 4*p/3 * depol_part.reshape(2**n_bits, 2**n_bits) 

# perform one depolarized XX layer (depol channel before and after XX gates)
def XX_depol_layer(rho, theta, p):
    XX = MS_XX(theta)

    # construct XX layer
    A = np.ones((int(np.log2(rho.size)/4),4,4)) * XX
    XX_layer = A[0]
    for ele in A[1:]:
        XX_layer = np.kron(XX_layer, ele)

    XX_layer_dag = XX_layer.conj().transpose()

    # depolarize before and after noiseless MS gate
    for i in range(int(np.log2(rho.size)/2)):
        rho = depol(rho, i, p/2)

    rho = XX_layer @ rho @ XX_layer_dag

    for i in range(int(np.log2(rho.size)/2)):
        rho = depol(rho, i, p/2)

    return rho


##### W GATES #####

# exact definition
def make_w(theta):
    return np.array(
        [
            [np.cos(theta - np.pi / 4), 0, 0, np.sin(theta - np.pi / 4)],
            [0, np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
            [0, np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
            [-np.sin(theta - np.pi / 4), 0, 0, np.cos(theta - np.pi / 4)],
        ]
    )

# decomposed into native gates
def W_decomp(theta):
	return np.real( MS_XX(-pi/2) @ kron(Z_rot(theta - pi/2), Z_rot(theta)) @ MS_XX(pi/2) )

# native gates with noisy angle selection noise using Kraus operators
def W_decomp_var(theta, var):
    p = 1/2 * (1+np.exp(-var/2))
    K_m = [np.sqrt(p)*MS_XX(-pi/2), np.sqrt(1-p)*MS_XX(-3*pi/2)]   # Kraus ops of XX(-pi/2)
    K_n = [np.sqrt(p)*MS_XX(pi/2), np.sqrt(1-p)*MS_XX(-pi/2)]     # Kraus ops of XX(pi/2)

    # Return Kraus ops of W
    return [Q @ kron(Z_rot(theta - pi/2), Z_rot(theta)) @ R for Q in K_m for R in K_n]

# spontaneous bit flip noise
def W_decomp_depol_layer(rho, theta, p):
    ZZ_layer = kron(Z_rot(theta - pi/2), Z_rot(theta), Z_rot(theta - pi/2), Z_rot(theta), Z_rot(theta - pi/2), Z_rot(theta))

    rho = XX_depol_layer(rho, pi/2, p)
    rho = ZZ_layer @ rho @ ZZ_layer.conj().transpose()
    rho = XX_depol_layer(rho, -pi/2, p)

    return np.real(rho)


##### U GATES #####

# exact definition
def make_u(theta):
    return np.array(
        [
            [np.cos(theta), 0, 0, np.sin(theta)],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-np.sin(theta), 0, 0, np.cos(theta)],
        ]
    )

# decomposed into native gates
def U_decomp(theta):
	return np.real( MS_XX(-np.pi/2) @ kron(Z_rot(theta), Z_rot(theta)) @ MS_XX(np.pi/2) )

# native gates with noisy angle selection noise using Kraus operators
def U_decomp_var(theta, var):
    p = 1/2 * (1+np.exp(-var/2))
    K_m = [np.sqrt(p)*MS_XX(-pi/2), np.sqrt(1-p)*MS_XX(-3*pi/2)]   # Kraus ops of XX(-pi/2)
    K_n = [np.sqrt(p)*MS_XX(pi/2), np.sqrt(1-p)*MS_XX(-pi/2)]     # Kraus ops of XX(pi/2)

    # Get Kraus ops of U
    K_s = [Q @ kron(Z_rot(theta), Z_rot(theta)) @ R for Q in K_m for R in K_n]

    return K_s

# spontaneous bit flip noise
def U_decomp_depol_layer(rho, theta, p):
    ZZ_layer = kron(Z_rot(theta), Z_rot(theta), Z_rot(theta), Z_rot(theta))

    rho = XX_depol_layer(rho, pi/2, p)
    rho = ZZ_layer @ rho @ ZZ_layer.conj().transpose()
    rho = XX_depol_layer(rho, -pi/2, p)

    return np.real(rho)

