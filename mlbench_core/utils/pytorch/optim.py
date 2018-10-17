import torch.optim as optim


class Optimizer(object):
    @staticmethod
    def create(config, model):
        lr = config.lr if config.lr else config.lr_per_sample * config.batch_size

        if config.optim == 'sgd':
            optimizer = optim.SGD(model.parameters(),
                                  lr=lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay,
                                  nesterov=config.nesterov)
        else:
            raise NotImplementedError("The optimizer `{}` specified by `config` is not implemented."
                                      .format(config.optim))

        return optimizer
