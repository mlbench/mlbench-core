from .resnet import get_resnet_model

__all__ = ['Models']


class Models(object):
    @staticmethod
    def create(config):
        if 'resnet' in config.model:
            return get_resnet_model(config)
        raise NotImplementedError
