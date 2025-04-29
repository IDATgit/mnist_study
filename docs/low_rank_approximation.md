this is a survey of low rank approximation methods to be used for fisher information matrix.

# Setup - Low rank approxmation
given a matrix $A \in \mathbb{R}^{m \times n}$ using SVD: $A = U\Sigma V$
where $U \in \mathbb{R}^{m \times m}$, $\Sigma \in \mathbb{R}^{m \times n}$, and $V \in \mathbb{R}^{n \times n}$ we are interested in an approximation: $A \approx A_{est} = U_k\Sigma_k V_k^T$ where $U_k \in \mathbb{R}^{m \times k}$, $\Sigma_k \in \mathbb{R}^{k \times k}$, and $V_k \in \mathbb{R}^{n \times k}$ contain only the first $k$ columns/singular values of the original matrices. This gives us a rank-$k$ approximation of $A$. 
## random subspace methods
### Randomized SVD
Randomized SVD (RSVD) is an efficient algorithm for computing a low-rank approximation of a large matrix. The key idea is to use random sampling to identify a subspace that captures most of the action of the matrix, then project the matrix onto this subspace to obtain a smaller matrix that can be decomposed using classical SVD methods.

### Algorithm

Given a matrix $A \in \mathbb{R}^{m \times n}$ and target rank $k$:

1. **Random Projection**: Generate a random Gaussian matrix $\Omega \in \mathbb{R}^{n \times (k+p)}$ where $p$ is a small oversampling parameter.
2. **Form Sampling Matrix**: Compute $Y = A\Omega$ to obtain a matrix $Y \in \mathbb{R}^{m \times (k+p)}$ that captures the range of $A$.
3. **QR Factorization**: Compute $Y = QR$ where $Q \in \mathbb{R}^{m \times (k+p)}$ has orthonormal columns
4. **Project Matrix**: Form $B = Q^T A \in \mathbb{R}^{(k+p) \times n}$
5. **SVD of Small Matrix**: Compute SVD of $B$ as $B = \hat{U}\hat{\Sigma}V^T$
6. **Recover Left Singular Vectors**: Set $U = Q\hat{U}$

The resulting approximation is $A \approx U\Sigma V^T$, where we typically keep only the first $k$ columns/values.

### Advantages

- Significantly faster than full SVD for large matrices
- Memory efficient as it avoids forming the full SVD
- Provides accuracy guarantees with high probability
- Well-suited for matrices with rapidly decaying singular values
- Can be implemented in a streaming fashion for very large datasets

The computational complexity is approximately $O(mn\log(k))$ compared to $O(mn\min(m,n))$ for full SVD, making it practical for analyzing large-scale Fisher Information Matrices in neural networks.


