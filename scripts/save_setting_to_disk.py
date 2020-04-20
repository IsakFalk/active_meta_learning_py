"""Create setting and save it to disk"""
from pathlib import Path
import logging
import argparse
from collections import OrderedDict


import numpy as np
import hickle as hkl


from active_meta_learning.data import (
    EnvironmentDataSet,
    GaussianNoiseMixture,
    HypercubeWithKVertexGaussian,
    UniformSphere,
    VonMisesFisherMixture,
)
from active_meta_learning.project_parameters import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


def create_vmf_params(d, k, kappa):
    mus = np.random.randn(k, d)
    mus /= np.linalg.norm(mus, axis=1).reshape(-1, 1)
    kappas = np.ones(k) * kappa
    ps = np.ones(k) / k
    return mus, kappas, ps


def stringify_parameter_dictionary(d, joiner="-", prefix="", suffix=""):
    l = []
    for key, val in d.items():
        if type(val) == float:
            l.append("{!s}={:.8g}".format(key, val))
        elif type(val) == int:
            l.append("{!s}={}".format(key, val))
        else:
            l.append("{!s}={}".format(key, val))
    output_string = prefix + joiner.join(l) + suffix
    return output_string


if __name__ == "__main__":
    # Available settings
    settings = [
        "hypersphere",
        "hypercube",
        "vmf_mixture",
    ]
    parser = argparse.ArgumentParser(description="Create setting and save to disk")
    parser.add_argument(
        "--setting", type=str, choices=settings, help="setting to construct"
    )
    parser.add_argument("-d", "--dimensions", type=int, help="dimension of setting")
    parser.add_argument(
        "-k", "--k_mixtures", type=int, help="number of mixtures (if any)"
    )
    parser.add_argument(
        "--w_sigma2",
        type=float,
        help="w ~ first2moments(m_i, w_sigma2*I) for each mixture i, if no mixtures: do nothing",
        default=0.0,
    )
    parser.add_argument(
        "--normalise_variance",
        help="normalise w variance by dimension",
        action="store_false",  # The default is True
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="where to save resulting setting object as .hkl file",
        default=str((DATA_DIR / "settings").resolve()),
    )

    args = parser.parse_args()
    d = args.dimensions
    k = args.k_mixtures
    original_w_s2 = args.w_sigma2
    if args.normalise_variance:
        w_s2 = original_w_s2 / d
    save_path = Path(args.save_path)

    if args.setting == "hypersphere":
        env = UniformSphere(d)
        params = OrderedDict(d=d)
        hkl.dump(
            env,
            save_path
            / stringify_parameter_dictionary(
                params, prefix="hypersphere-", suffix=".hkl"
            ),
        )
    elif args.setting == "hypercube":
        env = HypercubeWithKVertexGaussian(d, k=k, s2=w_s2)
        params = OrderedDict(
            d=d, k=k, w_s2=w_s2, normalise_variance=args.normalise_variance
        )
        hkl.dump(
            env,
            save_path
            / stringify_parameter_dictionary(
                params, prefix="hypercube-", suffix=".hkl"
            ),
        )
    elif args.setting == "vmf_mixture":
        kappa = 1.0 / w_s2
        mus, kappas, ps = create_vmf_params(d, k, kappa)
        env = VonMisesFisherMixture(mus, kappas, ps)
        params = OrderedDict(
            d=d, k=k, kappa=kappa, normalise_variance=args.normalise_variance
        )
        hkl.dump(
            env,
            save_path
            / stringify_parameter_dictionary(
                params, prefix="vmf_mixture-", suffix=".hkl"
            ),
        )
    else:
        raise NotImplementedError("{} setting not implemented.".format(args.setting))
