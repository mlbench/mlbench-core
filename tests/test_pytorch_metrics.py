import numpy as np
import torch

from mlbench_core.evaluation.pytorch.metrics import (
    BLEUScore,
    DiceCoefficient,
    F1Score,
    Perplexity,
    TopKAccuracy,
)


def test_f1_score():
    output = torch.tensor([1, 1, 1, 1, 1]).reshape(5, 1)
    target = torch.tensor([0, 0, 0, 0, 0]).reshape(5, 1)

    f1 = F1Score()
    score = f1(output, target)

    assert score.item() == 0

    output = torch.tensor([1, 1, 1, 0, 1]).reshape(5, 1)
    target = torch.tensor([1, 0, 1, 1, 0]).reshape(5, 1)

    precision = 2 / (2 + 2)
    recall = 2 / (2 + 1)

    score = f1(output, target)
    expected_score = 2 * (precision * recall) / (precision + recall)
    np.testing.assert_almost_equal(score.item(), expected_score)


def test_top1_accuracy():
    output_1 = torch.tensor([[0, 1], [0, 1], [1, 0], [0, 1], [1, 0]]).reshape(5, 2)
    output_2 = torch.tensor([1, 1, 0, 1, 0]).reshape(5, 1)
    target = torch.tensor([0, 1, 0, 0, 1]).reshape(5, 1)

    acc = TopKAccuracy(topk=1)
    expected_score = (2 / 5) * 100

    actual_score_1 = acc(output_1, target)
    actual_score_2 = acc(output_2, target)

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

    actual_score_1 = acc(output_1, target)

    assert actual_score_1 == expected_score


def test_perplexity():
    target = torch.randint(high=1000, size=(100, 1))
    outputs = torch.randn((100, 1000, 1))

    true_ppl = torch.exp(torch.nn.functional.cross_entropy(outputs, target))
    ppl = Perplexity()
    ppl_score = ppl(outputs, target)

    assert ppl_score == true_ppl


def test_dice_coefficient():
    target = torch.Tensor([1, 1, 1, 0, 0, 1]).view(-1, 1)
    output = torch.Tensor([0.2, 0.6, 0.1, 0.15, 0.1, 0.8]).view(-1, 1)

    dice = DiceCoefficient()

    loss = dice(output, target).item()

    assert round(loss, 1) == 0.6


def test_raw_bleu_score():
    outputs = ["the quick yellow fox jumps over the active dog"]
    target = ["the quick brown fox jumps over the lazy dog"]

    bl = BLEUScore(use_raw=True)
    score = bl(outputs, target)

    assert round(score.item(), 1) == 36.9
