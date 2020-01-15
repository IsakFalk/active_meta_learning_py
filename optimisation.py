import numpy as np
from tqdm import tqdm


def rho_t_func_kh(t):
    """Function for rho_t in FW to get KH"""
    return 1 / (1 + t)


class KernelHerding:
    """
    Notation, we let Qt denote the set of sampled points after finishing iteration t,
    Ut denote the set of unsampled points after finishing iteration t. K_ab then corresponds
    to the kernel matrix by taking the inner product \Phi_a^T \Phi_b in the RKHS.
    """

    def __init__(self, K, stop_t=None):
        self.K = K
        self.n = K.shape[0]
        self.stop_t = stop_t
        if self.stop_t is None:
            self.stop_t = self.n
        self.sampled_order = np.zeros(stop_t).astype(int)

        # These just help run the algorithm
        # These are all boolean, a one represents that
        # the x_i at that index has been sampled / unsampled respectively
        self.initial_indices = np.ones(self.n).astype(bool)
        self.sampled_indices = np.zeros(self.n).astype(bool)
        self.unsampled_indices = ~self.sampled_indices
        self.arange_indices = np.arange(0, self.n).astype(int)

        # Check for faults
        assert self.K.shape == (
            self.n,
            self.n,
        ), "K should be of shape ({}, {}), is of shape {}".format(
            self.n, self.n, self.K.shape
        )
        assert self.stop_t > 0, "stop_t needs to be positive"

        # Sanity check: keep objective values for each iteration
        self.objective_curve = np.zeros(self.stop_t)

    def _objective_func(self, t):
        """Calculate the objective function using sampled_indices until time t

        Note that this is a vectorised version of the one in the paper"""
        # This gets the corresponding sub-kernel matrices
        K_nUtm1 = self.K[np.ix_(self.initial_indices, self.unsampled_indices)]
        K_Qtm1Utm1 = self.K[np.ix_(self.sampled_indices, self.unsampled_indices)]

        # Original factor is T, but since we count from 0, need to increment by 1
        # Get it in the form of the FW algorithm to be able to compare
        J = (float(t) / self.n) * K_nUtm1.sum(axis=0) - K_Qtm1Utm1.sum(axis=0)
        assert J.shape == (
            self.unsampled_indices.sum(),
        ), "The output shape of J.shape should be ({},), is {} ".format(
            self.unsampled_indices.sum(), J.shape
        )
        J = np.atleast_1d(J.squeeze())
        return J

    def restart(self):
        """Start over, reinitialise everything"""
        self.sampled_order = np.zeros(self.stop_t).astype(int)
        self.initial_indices = np.ones(self.n).astype(bool)
        self.sampled_indices = np.zeros(self.n).astype(bool)
        self.unsampled_indices = ~self.sampled_indices
        self.arange_indices = np.arange(0, self.n).astype(int)
        self.objective_curve = np.zeros(self.stop_t)

    def run_kernel_herding(self):
        """Kernel herding on the empirical distribution of X through K

        Run kernel herding on the dataset (x_i)_i^n using the kernel matrix
        K represented as a numpy array of shape (n, n), where K_ij = K(x_i, x_j).
        Since the herding algorithm gives a new ordering of the dataset corresponding
        to what datapoint to include when we only return the indices of the new ordering.
        This is an array called return_order such that return_order[t] = index of x returned at
        end of t'th iteration of kernel herding.

        :param K: (np.ndarray, (n, n)) kernel matrix
        :param stop_t: (int, >0) stop running when t >= stop_t

        :return sampled_order: (np.array, (stop_t,)) the returned indices in the dataset for each t
        """
        self.restart()

        # Initially (t=0) we sample x_0 always
        self.sampled_order[0] = 0
        self.sampled_indices[0] = True
        self.unsampled_indices = ~self.sampled_indices
        self.objective_curve[0] = np.nan

        for t in tqdm(range(1, self.stop_t)):
            # The objective function of all points we can sample
            J = self._objective_func(t)
            # Get the index for the argmax
            J_argmax = J.argmax()
            # The index is not correct as we removed all indices
            # which has been sampled, so we map back to the correct index
            map_back_to_correct_index = self.arange_indices[self.unsampled_indices]
            sampled_index_at_t = map_back_to_correct_index[J_argmax]
            # Update the index arrays
            self.sampled_order[t] = sampled_index_at_t
            self.sampled_indices[sampled_index_at_t] = True
            self.unsampled_indices = ~self.sampled_indices
            # Put objective value of t
            self.objective_curve[t] = J.max()

    def run(self):
        """Run the algorithm, consistent interface for calling self.run_{algorithm}() to self.run()"""
        self.run_kernel_herding()


class FrankWolfe:
    """
    Notation, we let Qt denote the set of sampled points after finishing iteration t,
    Ut denote the set of unsampled points after finishing iteration t. K_ab then corresponds
    to the kernel matrix by taking the inner product \Phi_a^T \Phi_b in the RKHS.
    """

    def __init__(self, K, stop_t=None, rho_t_func=rho_t_func_kh):
        self.K = K
        self.n = K.shape[0]
        self.stop_t = stop_t
        if self.stop_t is None:
            self.stop_t = self.n
        self.rho_t_func = rho_t_func
        # self.W is such that self.W[t, j] = w_j at t
        # where j is the index of the sampled x at time j
        self.W = np.zeros((self.stop_t, self.stop_t))
        self.sampled_order = np.zeros(stop_t).astype(int)

        # These just help run the algorithm
        # These are all boolean, a one represents that
        # the x_i at that index has been sampled / unsampled respectively
        self.initial_indices = np.ones(self.n).astype(bool)
        self.sampled_indices = np.zeros(self.n).astype(bool)
        self.unsampled_indices = ~self.sampled_indices
        self.arange_indices = np.arange(0, self.n).astype(int)

        # Check for faults
        assert self.K.shape == (
            self.n,
            self.n,
        ), "K should be of shape ({}, {}), is of shape {}".format(
            self.n, self.n, self.K.shape
        )
        assert self.stop_t > 0, "stop_t needs to be positive"

        # Sanity check: keep objective values for each iteration
        self.objective_curve = np.zeros(self.stop_t)

    def _update_W(self, t):
        """We use the following observation

        Following Bach the w_y of the chosen instances
        up until time t, follows the recursive structure of
        w_t: w_t[t] = rho_t, w_t[u] = 0, u > t and for the rest
        w_{t+1} = w_{t} * (1 - rho_t) + [0, ..., rho_t, 0, ..., 0]
        where rho_t is in the t'th place
        """
        w_tm1 = self.W[t - 1, :]
        rho_t = self.rho_t_func(t)
        rho_t_indexer = np.zeros(self.n)
        # NB: in Bach paper, we would multiply by rho_t(t-1)
        # but since we count from 0, we instead multiply by rho_t(t)
        rho_t_indexer[t] = rho_t
        w_t = w_tm1 * (1 - rho_t) + rho_t_indexer

        self.W[t, :] = w_t

    def _objective_func(self, t):
        """Calculate the objective function using sampled_indices until time t

        Note that this is a vectorised version of the one in the paper"""
        # This gets the corresponding sub-kernel matrices
        K_nUtm1 = self.K[np.ix_(self.initial_indices, self.unsampled_indices)]
        K_Qtm1Utm1 = self.K[np.ix_(self.sampled_indices, self.unsampled_indices)]

        assert K_nUtm1.shape == (
            self.n,
            self.n - t,
        ), "K_nUtm1 should have shape ({}, {}), has shape {}".format(
            self.n, t, K_nUtm1.shape
        )
        assert K_Qtm1Utm1.shape == (
            t,
            self.n - t,
        ), "K_Qtm1Utm1 should have shape ({}, {}), has shape {}".format(
            t, self.n - t, K_Qtm1Utm1.shape
        )

        # At time t, we need to get the previous weights, up til diagonal element
        w_tm1 = self.W[t - 1, :t].reshape(-1, 1)
        uniform_wn = np.ones((self.n, 1)) / self.n

        weighted_sum = w_tm1.T @ K_Qtm1Utm1
        repulsive_sum = uniform_wn.T @ K_nUtm1

        assert weighted_sum.shape == (
            1,
            self.n - t,
        ), "Number of elements in weigted_sum at {} should be {}, is {}".format(
            t, self.n - t, weighted_sum.shape[0]
        )
        assert repulsive_sum.shape == (
            1,
            self.n - t,
        ), "Number of elements in repulsive_sum at {} should be {}, is {}".format(
            t, self.n - t, repulsive_sum.shape[0]
        )

        J = weighted_sum - repulsive_sum
        assert J.shape == (
            1,
            self.unsampled_indices.sum(),
        ), "The output shape of J.shape should be ({},), is {} ".format(
            self.unsampled_indices.sum(), J.shape
        )
        J = np.atleast_1d(J.squeeze())
        return J

    def restart(self):
        """Start over, reinitialise everything"""
        self.W = np.zeros((self.stop_t, self.stop_t))
        self.sampled_order = np.zeros(self.stop_t).astype(int)
        self.initial_indices = np.ones(self.n).astype(bool)
        self.sampled_indices = np.zeros(self.n).astype(bool)
        self.unsampled_indices = ~self.sampled_indices
        self.arange_indices = np.arange(0, self.n).astype(int)
        self.objective_curve = np.zeros(self.stop_t)

    def run_frank_wolfe(self):
        """Kernel herding on the empirical distribution of X through K

        Run kernel herding on the dataset (x_i)_i^n using the kernel matrix
        K represented as a numpy array of shape (n, n), where K_ij = K(x_i, x_j).
        Since the herding algorithm gives a new ordering of the dataset corresponding
        to what datapoint to include when we only return the indices of the new ordering.
        This is an array called return_order such that return_order[t] = index of x returned at
        end of t'th iteration of kernel herding.

        :param K: (np.ndarray, (n, n)) kernel matrix
        :param stop_t: (int, >0) stop running when t >= stop_t

        :return sampled_order: (np.array, (stop_t,)) the returned indices in the dataset for each t
        """
        self.restart()

        # Initially (t=0) we sample x_0 always
        self.sampled_order[0] = 0
        self.sampled_indices[0] = True
        self.unsampled_indices = ~self.sampled_indices
        self.W[0, 0] = 1.0
        self.objective_curve[0] = np.nan

        for t in tqdm(range(1, self.stop_t)):
            # The objective function of all points we can sample
            J = self._objective_func(t)
            # Get the index for the argmin
            J_argmin = J.argmin()
            # The index is not correct as we removed all indices
            # which has been sampled, so we map back to the correct index
            map_back_to_correct_index = self.arange_indices[self.unsampled_indices]
            sampled_index_at_t = map_back_to_correct_index[J_argmin]
            # Update the index arrays
            self.sampled_order[t] = sampled_index_at_t
            self.sampled_indices[sampled_index_at_t] = True
            self.unsampled_indices = ~self.sampled_indices
            self._update_W(t)
            # Put objective value of t
            self.objective_curve[t] = J.min()

    def run(self):
        """Run the algorithm, consistent interface for calling self.run_{algorithm}() to self.run()"""
        self.run_frank_wolfe()


class FrankWolfeLineSearch:
    def __init__(self, K, stop_t=None):
        self.K = K
        self.n = K.shape[0]
        self.stop_t = stop_t
        if self.stop_t is None:
            self.stop_t = self.n
        # self.W is such that self.W[t, j] = w_j at t
        # where j is the index of the sampled x at time j
        self.W = np.zeros((self.stop_t, self.stop_t))
        self.sampled_order = np.zeros(stop_t).astype(int)

        # These just help run the algorithm
        # These are all boolean, a one represents that
        # the x_i at that index has been sampled / unsampled respectively
        self.initial_indices = np.ones(self.n).astype(bool)
        self.sampled_indices = np.zeros(self.n).astype(bool)
        self.unsampled_indices = ~self.sampled_indices
        self.arange_indices = np.arange(0, self.n).astype(int)

        # Check for faults
        assert self.K.shape == (
            self.n,
            self.n,
        ), "K should be of shape ({}, {}), is of shape {}".format(
            self.n, self.n, self.K.shape
        )
        assert self.stop_t > 0, "stop_t needs to be positive"

        # Sanity check:
        # - keep objective values for each iteration
        # - keep rho_t to see the actual line-length
        self.objective_curve = np.zeros(self.stop_t)
        self.rho_t = np.zeros(self.stop_t)

    def calculate_rho(self, t):
        """Calculate rho using line-search"""
        # Let u = np.ones((n, 1)) / n,
        # subscript t, n, s indicates currently sampled datapoints, n all of them
        # and s the one sampled most recently
        uniform_wn = np.ones((self.n, 1)) / self.n
        w = self.W[t - 1, :t].reshape(t, 1)

        # Asserts
        assert uniform_wn.shape == (
            self.n,
            1,
        ), "uniform_wn should have shape (n, 1), has shape {}".format(uniform_wn.shape)
        assert np.allclose(
            uniform_wn.sum(), 1
        ), "sum(uniform_wn) should be equal to 1, is equal to {}".format(
            uniform_wn.sum()
        )
        assert w.shape == (t, 1), "w should have shape ({}, 1), has shape {}".format(
            t, w.shape
        )

        # Sub-matrices
        # t: index along sampled until start of time t
        # n: all indices
        # x: index of x chosen by fw-ls at time t
        x_index = np.array([self.sampled_order[t]])
        K_tt = self.K[np.ix_(self.sampled_order[:t], self.sampled_order[:t])]
        K_nt = self.K[np.ix_(self.arange_indices, self.sampled_order[:t])]
        K_nx = self.K[np.ix_(self.arange_indices, x_index)]
        K_tx = self.K[np.ix_(self.sampled_order[:t], x_index)]
        K_xx = self.K[self.sampled_order[t], self.sampled_order[t]]

        # First the numerator
        # w.T @ K_tt @ w - u.T @ K_nt @ w - w.T @ K_tx + u.T @ K_nx
        numerator = (
            w.T @ K_tt @ w - uniform_wn.T @ K_nt @ w - w.T @ K_tx + uniform_wn.T @ K_nx
        )
        # Then the denominator
        # K_ss - 2 * w.T @ K_ts + w.T @ K_tt @ w
        denominator = K_xx - 2 * w.T @ K_tx + w.T @ K_tt @ w

        assert numerator > 0, "Numerator needs to be positive, is {}".format(numerator)
        assert denominator > 0, "Denominator needs to be positive, is {}".format(
            denominator
        )

        rho_t = numerator / denominator

        assert (
            0 <= rho_t <= 1
        ), "rho_{} needs to be in [0, 1], but is {} at time {}".format(t, rho_t, t)

        return rho_t

    def _update_W(self, t):
        """We use the following observation

        Following Bach the w_y of the chosen instances
        up until time t, follows the recursive structure of
        w_t: w_t[t] = rho_t, w_t[u] = 0, u > t and for the rest
        w_{t+1} = w_{t} * (1 - rho_t) + [0, ..., rho_t, 0, ..., 0]
        where rho_t is in the t'th place
        """
        w_tm1 = self.W[t - 1, :]
        rho_t = self.calculate_rho(t)
        rho_t_indexer = np.zeros(self.n)
        # NB: in Bach paper, we would multiply by rho_t(t-1)
        # but since we count from 0, we instead multiply by rho_t(t)
        rho_t_indexer[t] = rho_t
        w_t = w_tm1 * (1 - rho_t) + rho_t_indexer

        self.W[t, :] = w_t

    def _objective_func(self, t):
        """Calculate the objective function using sampled_indices until time t

        Note that this is a vectorised version of the one in the paper"""
        # This gets the corresponding sub-kernel matrices
        K_nUtm1 = self.K[np.ix_(self.initial_indices, self.unsampled_indices)]
        K_Qtm1Utm1 = self.K[np.ix_(self.sampled_indices, self.unsampled_indices)]

        assert K_nUtm1.shape == (
            self.n,
            self.n - t,
        ), "K_nUtm1 should have shape ({}, {}), has shape {}".format(
            self.n, t, K_nUtm1.shape
        )
        assert K_Qtm1Utm1.shape == (
            t,
            self.n - t,
        ), "K_Qtm1Utm1 should have shape ({}, {}), has shape {}".format(
            t, self.n - t, K_Qtm1Utm1.shape
        )

        # At time t, we need to get the previous weights, up til diagonal element
        w_tm1 = self.W[t - 1, :t].reshape(-1, 1)
        uniform_wn = np.ones((self.n, 1)) / self.n

        weighted_sum = w_tm1.T @ K_Qtm1Utm1
        repulsive_sum = uniform_wn.T @ K_nUtm1

        assert weighted_sum.shape == (
            1,
            self.n - t,
        ), "Number of elements in weigted_sum at {} should be {}, is {}".format(
            t, self.n - t, weighted_sum.shape[0]
        )
        assert repulsive_sum.shape == (
            1,
            self.n - t,
        ), "Number of elements in repulsive_sum at {} should be {}, is {}".format(
            t, self.n - t, repulsive_sum.shape[0]
        )

        J = weighted_sum - repulsive_sum
        assert J.shape == (
            1,
            self.unsampled_indices.sum(),
        ), "The output shape of J.shape should be ({},), is {} ".format(
            self.unsampled_indices.sum(), J.shape
        )
        J = np.atleast_1d(J.squeeze())
        return J

    def restart(self):
        """Start over, reinitialise everything"""
        self.W = np.zeros((self.stop_t, self.stop_t))
        self.sampled_order = np.zeros(self.stop_t).astype(int)
        self.initial_indices = np.ones(self.n).astype(bool)
        self.sampled_indices = np.zeros(self.n).astype(bool)
        self.unsampled_indices = ~self.sampled_indices
        self.arange_indices = np.arange(0, self.n).astype(int)
        self.objective_curve = np.zeros(self.stop_t)

    def run_frank_wolfe(self):
        """Kernel herding on the empirical distribution of X through K

        Run kernel herding on the dataset (x_i)_i^n using the kernel matrix
        K represented as a numpy array of shape (n, n), where K_ij = K(x_i, x_j).
        Since the herding algorithm gives a new ordering of the dataset corresponding
        to what datapoint to include when we only return the indices of the new ordering.
        This is an array called return_order such that return_order[t] = index of x returned at
        end of t'th iteration of kernel herding.

        :param K: (np.ndarray, (n, n)) kernel matrix
        :param stop_t: (int, >0) stop running when t >= stop_t

        :return sampled_order: (np.array, (stop_t,)) the returned indices in the dataset for each t
        """
        self.restart()

        # Initially (t=0) we sample x_0 always
        self.sampled_order[0] = 0
        self.sampled_indices[0] = True
        self.unsampled_indices = ~self.sampled_indices
        self.W[0, 0] = 1.0
        self.objective_curve[0] = np.nan
        self.rho_t[0] = np.nan

        for t in tqdm(range(1, self.stop_t)):
            # The objective function of all points we can sample
            J = self._objective_func(t)
            # Get the index for the argmin
            J_argmin = J.argmin()
            # The index is not correct as we removed all indices
            # which has been sampled, so we map back to the correct index
            map_back_to_correct_index = self.arange_indices[self.unsampled_indices]
            sampled_index_at_t = map_back_to_correct_index[J_argmin]
            # Update the index arrays
            self.sampled_order[t] = sampled_index_at_t
            self.sampled_indices[sampled_index_at_t] = True
            self.unsampled_indices = ~self.sampled_indices
            self._update_W(t)
            # Put objective value of t
            self.objective_curve[t] = J.min()
            # Put rho_t
            self.rho_t[t] = self.calculate_rho(t)

    def run(self):
        """Run the algorithm, consistent interface for calling self.run_{algorithm}() to self.run()"""
        self.run_frank_wolfe()
