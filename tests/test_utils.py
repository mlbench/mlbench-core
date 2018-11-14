#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.utils.pytorch.helpers` package."""

import pytest

from mlbench_core.utils import Tracker


def test_tracker():
    tracker = Tracker()

    assert tracker is not None