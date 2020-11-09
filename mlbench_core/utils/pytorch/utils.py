import torch


@torch.jit.script
def orthogonalize(matrix, eps=torch.FloatTensor([1e-16])):
    """Function used to orthogonalize a matrix.

    Args:
        matrix (torch.Tensor): Matrix to orthogonalize
        eps (torch.FloatTensor): Used to avoid division by zero (default: 1e-16)
    """
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


def pack_tensors(tensors, use_cuda=False):
    """
    Packs a list of tensors into one 1-dimensional tensor.

    Args:
        tensors (list[torch.Tensor]): The tensors to pack
        use_cuda (bool): Whether the resulting tensor should be on cuda

    Returns:
        (torch.Tensor, list[int], list[(int, int)]):
            The flattened tensors, the list start indices of each packed tensor,
            and the original shape of each tensor.

            Those values are used to then unpack the tensor
    """
    indices = [0]
    for tensor in tensors:
        new_end = indices[-1] + tensor.nelement()
        indices.append(new_end)

    tensor_sizes = [t.size() for t in tensors]

    vec = torch.empty(
        indices[-1],
        device=tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu",
        dtype=tensors[0].dtype,
    )

    for tensor, start_idx, end_idx in zip(tensors, indices[:-1], indices[1:]):
        vec[start_idx:end_idx] = tensor.data.view(-1)

    return vec, indices, tensor_sizes


def unpack_tensors(aggregated, indices, sizes):
    """
    Unpacks a 1-dimensional tensor into a list of tensors

    Args:
        aggregated (torch.Tensor): The 1-dimensional tensor
        indices (List[Int]): The start index of each tensor
        sizes (List[(Int, Int)]): The size of each resulting tensor

    Returns:
        List[torch.Tensor]: The unpacked tensors
    """
    start_index = indices[:-1]
    end_index = indices[1:]

    tensors = []
    for i, (start, end) in enumerate(zip(start_index, end_index)):
        tensors.append(aggregated[start:end].view(sizes[i]))

    return tensors
