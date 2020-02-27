import torch
from mlbench_core.evaluation.pytorch.metrics import *
import numpy as np

def test_f1_score():
    output = torch.tensor([1, 1, 1, 1, 1]).reshape(5, 1)
    target = torch.tensor([0, 0, 0, 0, 0]).reshape(5, 1)

    f1 = F1Score()
    score = f1(None, output, target)

    assert score.item() == 0

    output = torch.tensor([1, 1, 1, 0, 1]).reshape(5, 1)
    target = torch.tensor([1, 0, 1, 1, 0]).reshape(5, 1)

    precision = 2 / (2 + 2)
    recall = 2 / (2 + 1)

    score = f1(None, output, target)
    expected_score = 2 * (precision * recall) / (precision + recall)
    np.testing.assert_almost_equal(score.item(), expected_score)
