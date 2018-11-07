#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.utils.pytorch.helpers` package."""

import pytest
import time

from mlbench_core.utils.pytorch.helpers import Timeit




def test_timeit():
    timeit = Timeit()

    cumu = timeit.cumu

    time.sleep(0.1)
    timeit.pause()
    new_cumu = timeit.cumu

    assert new_cumu - cumu - 0.1 < 0.01

    time.sleep(0.1)
    newer_cumu = timeit.cumu

    assert new_cumu == newer_cumu

    timeit.resume()
    time.sleep(0.1)
    timeit.pause()

    last_cumu = timeit.cumu

    assert last_cumu - newer_cumu - 0.1 < 0.01
    assert last_cumu - cumu - 0.3 < 0.01


