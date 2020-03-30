import random

import numpy as np
import torch
from spherecluster.util import sample_vMF
from torch.utils.data import IterableDataset

################
# Environments #
################


class VonMisesFisherMixture:
    def __init__(self, mus, kappas, ps):
        assert mus.shape[0] == kappas.shape[0]
        assert all(kappas >= 0)
        assert all(ps >= 0 - 1e-6)
        self.mus = mus
        self.k, self.d = self.mus.shape
        self.kappas = kappas
        self.ps = ps / ps.sum()  # normalise to get probability

    def sample(self, n):
        assert type(n) == int
        assert n > 0
        W = np.zeros((n, self.d))
        for i in range(n):
            cluster = np.random.choice(self.k, p=self.ps)
            W[i, :] = sample_vMF(
                self.mus[cluster, :], self.kappas[cluster], num_samples=1
            )
        return W


class UniformSphere:
    def __init__(self, d):
        self.d = d

    def sample(self, n):
        assert type(n) == int
        assert n > 0
        W = np.zeros((n, self.d))
        for i in range(n):
            w = np.random.randn(self.d)
            w /= np.linalg.norm(w)
            W[i, :] = w
        return W


class UniformHypercube:
    def __init__(self, d):
        self.d = d

    def sample(self, n):
        assert type(n) == int
        assert n > 0
        W = np.random.rand(n, self.d)
        return W


# Deprecated in favour of HypercubeWithAllVertexGaussian
class HypercubeWithVertexGaussian:
    def __init__(self, d, s2):
        self.s2 = s2
        self.d = d

    def _sample_mixture(self):
        """Each vertex is in {0, 1}^d"""
        return np.random.randint(2, size=self.d).reshape(-1, self.d)

    def sample(self, n):
        assert type(n) == int
        assert n > 0
        W = np.zeros((n, self.d))
        for i in range(n):
            mu = self._sample_mixture()
            W[i, :] = mu + self.s2 * np.random.randn(self.d)
        return W


class HypercubeWithAllVertexGaussian:
    def __init__(self, d, s2):
        self.s2 = s2
        self.d = d

    def _sample_mixture(self):
        """Each vertex is in {0, 1}^d"""
        return np.random.randint(2, size=self.d).reshape(-1, self.d)

    def sample(self, n):
        assert type(n) == int
        assert n > 0
        W = np.zeros((n, self.d))
        for i in range(n):
            mu = self._sample_mixture()
            W[i, :] = mu + self.s2 * np.random.randn(self.d)
        return W


class HypercubeWithKVertexGaussian:
    def __init__(self, d, k, s2, mixture_vertices=None):
        self.d = d
        self.k = k
        self.s2 = s2
        if mixture_vertices is None:
            self._generate_mixture_vertices()
        else:
            assert type(mixture_vertices) == list
            assert len(mixture_vertices) == self.k
            self.mixture_vertices = mixture_vertices

    def _generate_mixture_vertices(self):
        num_mixtures = 0
        mixture_vertices = []
        while num_mixtures < self.k:
            vertex = np.random.randint(2, size=self.d).reshape(-1, self.d).tolist()
            if vertex not in mixture_vertices:
                mixture_vertices.append(vertex)
                num_mixtures += 1
        mixture_vertices = [np.array(ll) for ll in mixture_vertices]
        self.mixture_vertices = mixture_vertices

    def _sample_mixture(self):
        return random.choice(self.mixture_vertices)

    def sample(self, n):
        assert type(n) == int
        assert n > 0
        W = np.zeros((n, self.d))
        for i in range(n):
            mu = self._sample_mixture()
            W[i, :] = mu + self.s2 * np.random.randn(self.d)
        return W


#############################
# Environment (Dataloaders) #
#############################


class GaussianNoiseMixture:
    def __init__(self, p, mus, s2s, keep_mixture_history=True):
        """Generate noise from mixture of Gaussian distributions

        p is probability of picking first mixture, mus and s2s arrays
        with entries being the mean and variance of each mixture"""
        np.testing.assert_allclose(p.sum(), 1.0)
        assert mus.shape == s2s.shape
        self.p = p
        self.mus = mus
        self.s2s = s2s
        # Keep sampled sequence of mixture as an array of integers
        self.keep_mixture_history = keep_mixture_history
        self.mixture_history = []

    def sample(self, task_w, X, Y):
        k = Y.shape[0]
        mixture = np.random.choice(np.arange(len(self.mus)), p=self.p)
        if self.keep_mixture_history:
            self.mixture_history.append(mixture)
        mu = self.mus[mixture]
        s2 = self.s2s[mixture]
        error = mu + np.sqrt(s2) * np.random.randn(k)
        error = error.reshape(-1, 1)
        return error

    def reset_history(self):
        self.mixture_history = []


class EnvironmentDataSet(IterableDataset):
    def __init__(self, k_shot, k_query, env, noise_w=0.01, noise_y=0.01):
        self.k_shot = k_shot
        self.k_query = k_query
        self.k_total = k_shot + k_query
        self.env = env
        self.noise_w = noise_w
        self.noise_y = noise_y
        self.d = self.env.d

    def _sample_task(self):
        while True:
            task_w = self.env.sample(n=1).reshape(-1, 1) + np.sqrt(
                self.noise_w
            ) * np.random.randn(self.d).reshape(-1, 1)
            assert task_w.shape == (self.d, 1), "{}".format(task_w.shape)
            X = np.random.rand(self.k_total, self.d)
            Y = X @ task_w
            # Allow noise to be a function, noise takes as input
            # w, X, Y
            if callable(self.noise_y):
                error = self.noise_y(task_w, X, Y).reshape(-1, 1)
                assert error.shape == (
                    self.k_total,
                    1,
                ), "error have shape {}, not {}".format(error.shape, Y.shape)
                Y += error
            elif type(self.noise_y) == float:
                Y += np.sqrt(self.noise_y) * np.random.randn(self.k_total).reshape(
                    -1, 1
                )
            else:
                raise ()

            # Split
            X_train = X[: self.k_shot, :]
            Y_train = Y[: self.k_shot, :]
            X_test = X[self.k_shot :, :]
            Y_test = Y[self.k_shot :, :]
            yield X_train, Y_train, X_test, Y_test, task_w

    def __iter__(self):
        return self._sample_task()

    def collate_fn(self, data):
        """Create batches from list of tuples (X_tr, X_te, Y_tr, Y_te)"""
        train_xs = []
        train_ys = []
        test_xs = []
        test_ys = []
        for X_train, Y_train, X_test, Y_test, task_w in data:
            train_xs.append(torch.tensor(X_train))
            train_ys.append(torch.tensor(Y_train))
            test_xs.append(torch.tensor(X_test))
            test_ys.append(torch.tensor(Y_test))
        # stack the datasets along dimension 0
        train_x = torch.stack(train_xs, dim=0)
        train_y = torch.stack(train_ys, dim=0)
        test_x = torch.stack(test_xs, dim=0)
        test_y = torch.stack(test_ys, dim=0)
        collated_batch = {
            "train": (train_x, train_y),
            "test": (test_x, test_y),
            "w": task_w,
        }
        return collated_batch


#########################
# Environment (Helpers) #
#########################


def normalise(X):
    return X / np.sqrt(np.sum(X ** 2, axis=1).reshape(-1, 1))


def generate_random_mus(k, d):
    return normalise(np.random.randn(k, d))


def generate_random_kappas(k):
    return np.random.exponential(2.0, size=k)


def generate_mixture_params(k, d):
    mus = generate_random_mus(k, d)
    kappas = generate_random_kappas(k)
    return mus, kappas
