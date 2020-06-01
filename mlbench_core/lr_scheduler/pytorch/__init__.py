"""Scheduling Learning Rates.
"""

from .lr import (
    SQRTTimeDecayLR,
    TimeDecayLR,
    cyclical_learning_rates,
    multistep_learning_rates_with_warmup,
    triangular_learning_rates,
)
