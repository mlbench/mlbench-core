from .centralized import *
from .decentralized import *
from .optim import *

optimizers = {
    "sign_sgd": SignSGD,
    "sparsified_sgd": SparsifiedSGD,
    "centralized_sparsified_sgd": CentralizedSparsifiedSGD,
    "centralized_sgd": CentralizedSGD,
    "centralized_adam": CentralizedAdam,
    "power_sgd": PowerSGD,
    "decentralized_sgd": DecentralizedSGD,
}


def get_optimizer(optimizer, **kwargs):
    """Returns an object of the class specified with the argument `optimizer`.

    Args:
        optimizer (str): name of the optimizer
        **kwargs (dict, optional): additional optimizer-specific parameters. For the list of supported parameters
            for each optimizer, please look at its documentation.
    """
    return optimizers[optimizer](**kwargs)
