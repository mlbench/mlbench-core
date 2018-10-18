from .controlflow import TrainValidation

__all__ = ['get_controlflow']


def get_controlflow(config):
    if config.validation:
        return TrainValidation()

    raise NotImplementedError("Control flow not implemented.")
