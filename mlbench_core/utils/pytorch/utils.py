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
