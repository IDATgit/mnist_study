import numpy as np
from scipy.stats import ortho_group
n = 1000 # matrix dimension
p = 60 # number of non zero eigenvalues
k = p * 3 # number of random vectors

# generate sparse eigenvalues matrix
sigma = np.concatenate((np.ones(p), np.zeros(n-p)), axis=0)
SIGMA = np.diag(sigma)
U = ortho_group.rvs(n)
A = U @ SIGMA @ U.T


# generate k random vectors with unit norm (each vector is a column of v)
v = np.random.randn(n, k)
# Normalize each column to have unit norm
v = v / np.linalg.norm(v, axis=0, keepdims=True)
# print (np.diag(v.T @ v)) # debug normalization verification

# compute A @ v
v_sigma = A @ v # (n, k)
v_sigma_t = v_sigma.T # (k, n) # each row of v_sigma_t is a vector

# SVD
U, S, V = np.linalg.svd(v_sigma_t @ v_sigma)
S = S / k
# Normalize by sqrt(k) to get true eigenvalue estimates
normalized_eigenvalues = np.sqrt(S)
print(normalized_eigenvalues[:p + 2])




