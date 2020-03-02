import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


def gaussian_kernel_matrix(X, Y=None, s2=1.0):
    """
    X: [n, d] matrix, where n is the number of samples, d is the dimension
    s2: the standard deviation parameter, K(x, y) = np.exp(-np.norm(x - y)**2/(2*s2))
    """
    return rbf_kernel(X, Y, gamma=(0.5 / s2))


def median_heuristic(sqeuc_dist_mat):
    n = sqeuc_dist_mat.shape[0]
    cart_ind = np.triu_indices(n=n, k=1)
    view_ind = tuple(np.ravel_multi_index(cart_ind, (n, n)))
    return np.median(sqeuc_dist_mat[view_ind])


def mmd2(K_xx, K_yy, K_xy):
    """
    Calculate the mmd ** 2 between two empirical samples X and Y

    :param K_xx: (np.ndarray, (n, n)) kernel matrix constructed from X
    :param K_yy: (np.ndarray, (m, m)) kernel matrix constructed from Y
    :param K_xy: (np.ndarray, (n, m)) kernel matrix constructed from X (rows) and Y (cols)
    """
    n, m = K_xy.shape
    mmd2 = (
        (K_xx.sum() / (n ** 2)) + (K_yy.sum() / (m ** 2)) - 2 * (K_xy.sum() / (m * n))
    )
    return mmd2


# NOTE: the hardcoded s2 values are from one run of 300 tasks
def mmd2_matrix(A):
    assert len(A.shape) == 3
    m, n, d = A.shape
    assert d == 2
    # Get median of all arrays
    # hardcode this
    base_s2 = 18.65
    M2 = np.zeros((m, m))
    for i in range(m):
        for j in range(i):
            K_xx = gaussian_kernel_matrix(A[i], s2=base_s2)
            # K_xx = K_xx + np.eye(n)*eps
            K_yy = gaussian_kernel_matrix(A[j], s2=base_s2)
            # K_yy = K_yy + np.eye(n)*eps
            K_xy = gaussian_kernel_matrix(A[i], A[j], s2=base_s2)
            M2[i, j] = mmd2(K_xx, K_yy, K_xy)
    M2 = M2 + M2.T
    return M2


def gaussian_kernel_mmd2_matrix(A):
    assert len(A.shape) == 3
    m, n, d = A.shape
    assert d == 2
    M2 = mmd2_matrix(A)
    meta_s2 = 0.0497
    return np.exp(-0.5 * M2 / meta_s2)
