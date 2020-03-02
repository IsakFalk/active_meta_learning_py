import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import scipy.linalg as linalg

############
# Kernels  #
############


def median_heuristic(X, num_samples=1000):
    X_subsampled = X[np.random.permutation(np.arange(X.shape[0]))[:num_samples]]
    pairwise_sq_dists = squareform(pdist(X, "sqeuclidean"))
    return np.median(pairwise_sq_dists[np.tril_indices(num_samples, k=-1)])


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
        K = means_X @ means_Y.T
    else:
        means_X = X.mean(axis=1)  # (m, d)
        K = means_X @ means_X.T
    return K


## Below helps calculating the image feature map
def linear_regression_weight_matrix(X, Y, l=0.01):
    """Calculate the linear regression weight matrix W

    X is an (m, d) matrix and Y is an (m, c) one-hot encoded
    output matrix. Note that X might itself be the output of a
    feature map.
    """
    assert X.shape[0] == Y.shape[0]
    m, d = X.shape
    _, c = Y.shape
    alpha = linalg.solve(X @ X.T + m * l * np.eye(m), Y, sym_pos=True)
    W = X.T @ alpha
    return W


def averaged_lin_reg_weights(X, Y, l=0.01):
    """Calculate the averaged kernel of the linear regression weight W

    The output is the mean over c and d"""
    return linear_regression_weight_matrix(X, Y, l).mean(axis=1)


def lin_reg_feature_map(M_x, M_y, l=0.01):
    """Calculate the feature map phi_D

    phi_D is a matrix of shape (N, d) where N is the number of tasks and
    d is the dimension of the domain of x.
    :param M_x: matrix of shape (N, m, d), m is the number of points in a task (k_shot + k_query)
    :param M_y: matrix of shape (N, m, c), c is the number of classes (or n_way)"""
    assert M_y.shape[:2] == M_x.shape[:2]
    N, m, d = M_x.shape
    _, _, c = M_y.shape
    phi_D = np.zeros((N, d))
    for i in range(N):
        phi_D[i, :] = averaged_lin_reg_weights(M_x[i, :], M_y[i, :], l)
    return phi_D


def lin_reg_gaussian_kernel_matrix(
    M_x, M_y, s2=1.0, l=0.01, use_median_heuristic=True, n_median=1000
):
    phi_D = lin_reg_feature_map(M_x, M_y, l)
    if use_median_heuristic:
        n_median = min(n_median, phi_D.shape[0])
        s2 = median_heuristic(phi_D, n_median)
    return gaussian_kernel_matrix(X=phi_D, s2=s2)


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
    mmd2 += 2 * np.finfo(float).eps
    assert mmd2 > 0.0, "mmd2 should be non-negative, is {}".format(mmd2)

    return mmd2
