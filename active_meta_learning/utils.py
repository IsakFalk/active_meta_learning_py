import random

import numpy as np
import torch
import torch.functional as F

from .kernels import mmd2

##########
# Torch  #
##########


def swish(x):
    return x * F.sigmoid(x)


####################
# Reproducibility  #
####################


def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    # if (seed is not None) and cudnn:
    #     torch.backends.cudnn.deterministic = True


############################
# Plotting and computation #
############################


def mmd2_curve(K, order):
    all_idx = np.arange(K.shape[0])
    num_steps = len(order)
    J = np.zeros(num_steps)
    for i in range(num_steps):
        K_yy = K[np.ix_(order[: i + 1], order[: i + 1])]
        K_xy = K[np.ix_(all_idx, order[: i + 1])]
        J[i] = mmd2(K, K_yy, K_xy)
    return J


##########
# Saving #
##########


def stringify_parameter_dictionary(d, joiner="-"):
    l = []
    for key, val in d.items():
        if type(val) == float:
            l.append("{!s}={:.2f}".format(key, val))
        elif type(val) == int:
            l.append("{!s}={}".format(key, val))
        else:
            l.append("{!s}={}".format(key, val))
    return joiner.join(l)
