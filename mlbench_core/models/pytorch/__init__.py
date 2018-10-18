from .resnet import get_resnet_model

__all__ = ['get_model']


def get_model(config):
    if 'resnet' in config.model:
        return get_resnet_model(config)
    raise NotImplementedError
