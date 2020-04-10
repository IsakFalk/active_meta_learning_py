import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

####################
# Helper functions #
####################


def median_heuristic(sqeuc_dist_mat, n_subsamples=None):
    if n_subsamples is None:
        n = sqeuc_dist_mat.shape[0]
        cart_ind = np.triu_indices(n=n, k=1)
    return np.median(sqeuc_dist_mat[cart_ind])


############
# Kernels  #
############


def gaussian_kernel_matrix(X, Y=None, s2=1.0):
    """
    X: [n, d] matrix, where n is the number of samples, d is the dimension
    s2: the standard deviation parameter, K(x, y) = np.exp(-np.norm(x - y)**2/(2*s2))
    """
    # Get the matrix K_XY
    if type(Y) == np.ndarray or type(Y) == np.ndarray:
        pairwise_sq_dists = cdist(X, Y, "sqeuclidean")
        K = np.exp(-0.5 * pairwise_sq_dists / s2)
    # Get the matrix K_XX
    else:
        pairwise_sq_dists = squareform(pdist(X, "sqeuclidean"))
        K = np.exp(-0.5 * pairwise_sq_dists / s2)
    return K


########
# MMD  #
########


def mmd2(K_xx, K_yy, K_xy, w_x=None, w_y=None):
    """
    Calculate the mmd ** 2 between two empirical samples X and Y

    Allow for weighted sums, such that
    \mu_x = \sum_i w_xi K(x_i, )
    and
    \mu_y = \sum_j w_yj K(y_j, )

    :param K_xx: (np.ndarray, (n, n)) kernel matrix constructed from X
    :param K_yy: (np.ndarray, (m, m)) kernel matrix constructed from Y
    :param K_xy: (np.ndarray, (n, m)) kernel matrix constructed from X (rows) and Y (cols)
    :param w_x: (np.ndarray, (n, 1)) weights of datapoints in X, is a distribution
    :param w_y: (np.ndarray, (m, 1)) weights of datapoints in Y, is a distribution
    """
    n, m = K_xy.shape

    assert (
        n == K_xx.shape[0]
    ), "Shapes must conform between K_xx and K_xy, K_xx.shape == {}, K_xy.shape == {}".format(
        K_xx.shape, K_xy.shape
    )
    assert (
        m == K_yy.shape[0]
    ), "Shapes must conform between K_yy and K_xy, K_yy.shape == {}, K_xy.shape == {}".format(
        K_yy.shape, K_xy.shape
    )

    if isinstance(w_x, np.ndarray) and isinstance(w_y, np.ndarray):
        assert np.isclose(w_x.sum(), 1) and np.isclose(
            w_y.sum(), 1
        ), "w_x and w_y must sum to 1"
        assert w_x.shape == (n, 1) and w_y.shape == (
            m,
            1,
        ), "w_x and w_y must conform to K_xx and K_yy, have w_x.shape == {}, w_y.shape == {} and K_xx.shape == {}, K_yy == {}".format(
            w_x.shape, w_y.shape, K_xx.shape, K_yy.shape
        )
        assert (w_x >= 0).all(), "All entries of w_x should be greater than zero"
        assert (w_y >= 0).all(), "All entries of w_y should be greater than zero"
        mmd2 = w_x.T @ K_xx @ w_x - 2 * w_x.T @ K_xy @ w_y + w_y.T @ K_yy @ w_y
    else:
        mmd2 = (
            (K_xx.sum() / (n ** 2))
            + (K_yy.sum() / (m ** 2))
            - 2 * (K_xy.sum() / (m * n))
        )

    # Had problem with negative values on order of machine epsilon
    e = np.finfo(float).eps
    while mmd2 < 0.0:
        e *= 2
        mmd2 += 2 * e

    return mmd2


def mmd2_matrix(A, median_heuristic_n_subsamples=None):
    assert len(A.shape) == 3
    m, n, d = A.shape
    if median_heuristic_n_subsamples is None:
        vec_A = A.reshape(-1, d)
        pairwise_square_dists = squareform(pdist(vec_A, "sqeuclidean"))
    else:
        vec_A = A.reshape(-1, d)
        subsample_indices = np.random.permutation(vec_A.shape[0])[
            :median_heuristic_n_subsamples
        ]
        vec_A = vec_A[subsample_indices]
        pairwise_square_dists = squareform(pdist(vec_A, "sqeuclidean"))
    base_s2 = median_heuristic(pairwise_square_dists)
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


def gaussian_kernel_mmd2_matrix(A, median_heuristic_n_subsamples=None):
    # Need to check for batches
    assert A.ndim == 3 or A.ndim == 4
    if A.ndim == 3:
        M2 = mmd2_matrix(A, median_heuristic_n_subsamples)
        meta_s2 = median_heuristic(M2)
    if A.ndim == 4:
        b, m, n, d = A.shape
        M2 = mmd2_matrix(A.reshape(b * m, n, d), median_heuristic_n_subsamples)
        meta_s2 = median_heuristic(M2)
    return np.exp(-0.5 * M2 / meta_s2)


def mean_embedding_linear_kernel_matrix(X, Y=None):
    """
    Calculate the mean embedding of the metadataset X (cross with Y if given)

    Given that metadatasets X has shape (m, n*b, d) where
    m is the number of metainstances, n is the total size of each
    metainstance, b is the batches (so we get n*b instances in one batch of
    metainstances) and d is the dimensionality of the dataset, calculate the matrix
    K such that K_ij = <mu_Xi, mu_Xj>
    """
    if type(Y) == np.ndarray:
        means_Y = Y.mean(axis=1)
        means_X = X.mean(axis=1)
        K = means_X @ means_X
    else:
        means_X = X.mean(axis=1)  # (m, d)
        K = means_X @ means_X
    return K
