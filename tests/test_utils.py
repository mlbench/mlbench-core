#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.utils.pytorch.helpers` package."""

import pytest

from mlbench_core.utils import Tracker
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy


def test_tracker():
    tracker = Tracker([TopKAccuracy(5)], 1, 0)

    assert tracker is not None