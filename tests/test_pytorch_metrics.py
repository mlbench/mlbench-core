import numpy as np
import torch

from mlbench_core.evaluation.pytorch.metrics import F1Score, TopKAccuracy


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


def test_top1_accuracy():
    output_1 = torch.tensor([[0, 1], [0, 1], [1, 0], [0, 1], [1, 0]]).reshape(5, 2)
    output_2 = torch.tensor([1, 1, 0, 1, 0]).reshape(5, 1)
    target = torch.tensor([0, 1, 0, 0, 1]).reshape(5, 1)

    acc = TopKAccuracy(topk=1)
    expected_score = (2 / 5) * 100

    actual_score_1 = acc(None, output_1, target)
    actual_score_2 = acc(None, output_2, target)

    assert actual_score_1 == expected_score
    assert actual_score_2 == expected_score


def test_top3_accuracy():
    output_1 = torch.tensor(
        [
            [0.2, 0.2, 0.3, 0.1],
            [0.15, 0.2, 0.05, 0.6],
            [0.25, 0.3, 0.15, 0.3],
            [0.3, 0.1, 0.2, 0.2],
            [0.15, 0.15, 0.2, 0.5],
        ]
    ).reshape(5, 4)
    target = torch.tensor([3, 1, 0, 2, 1]).reshape(5, 1)

    acc = TopKAccuracy(topk=3)
    expected_score = (3 / 5) * 100

    actual_score_1 = acc(None, output_1, target)

    assert actual_score_1 == expected_score
