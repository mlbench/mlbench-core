import itertools


def maybe_range(maximum):
    """Map an integer or None to an integer iterator starting from 0 with stride 1.

    If maximum number of batches per epoch is limited, then return an finite
    iterator. Otherwise, return an iterator of infinite length.

    Args:
        maximum (int | None): Maximum number of steps in iterator.
            If none, returns iterator of infinite length

    Returns:
        (iterator)
    """
    if maximum is None:
        counter = itertools.count(0)
    else:
        counter = range(maximum)
    return counter


def convert_dtype(dtype, obj):
    """Converts given tensor to given dtype

    Args:
        dtype (str): One of `fp32` or `fp64`
        obj (`obj`:torch.Tensor | `obj`:torch.nn.Module): Module or tensor to convert

    Returns:
        (`obj`:torch.Tensor | `obj`:torch.nn.Module): Converted tensor or module
    """
    # The object should be a ``module`` or a ``tensor``
    if dtype == "fp32":
        return obj.float()
    elif dtype == "fp64":
        return obj.double()
    else:
        raise NotImplementedError("dtype {} not supported.".format(dtype))


def prepare_batch(data, target, dtype, transform_target_dtype=False, use_cuda=False):
    """Prepares a batch for training by changing the type and sending to cuda
    if necessary

    Args:
        data (`obj`:torch.Tensor): The input tensor
        target (`obj`:torch.Tensor): The target tensor
        dtype (str): One of `fp32` or `fp64`, data type to transform input and/or target
        transform_target_dtype (bool): Transform target to `dtype` too
        use_cuda (bool): Send tensors to GPU

    Returns:
        (`obj`:torch.Tensor, `obj`:torch.Tensor): Input and target tensors
    """
    data = convert_dtype(dtype, data)
    if transform_target_dtype:
        target = convert_dtype(dtype, target)

    if use_cuda:
        data, target = data.cuda(), target.cuda()

    return data, target


def iterate_dataloader(
    dataloader,
    dtype,
    max_batch_per_epoch=None,
    use_cuda=False,
    transform_target_type=False,
):
    """Function that returns an iterator on the given loader.
    Can be used to limit the number of batches, converting input and target dtypes
    and sending to GPU

    Args:
        dataloader (`obj`:torch.utils.data.DataLoader): The loader
        dtype (str): Type to convert to (`fp32` or `fp64`)
        max_batch_per_epoch (int | None): Maximum number of batches
        use_cuda (bool): Send tensors to GPU
        transform_target_type (bool): Transform target dtype as well

    Returns:
        (iterator): An iterator over the data
    """
    for _, (data, target) in zip(maybe_range(max_batch_per_epoch), dataloader):
        data, target = prepare_batch(
            data=data,
            target=target,
            dtype=dtype,
            transform_target_dtype=transform_target_type,
            use_cuda=use_cuda,
        )

        yield data, target
