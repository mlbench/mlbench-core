"""Scheduling Learning Rates.
"""

from .lr import (
    ExponentialWarmupMultiStepLR,
    LRLinearWarmUp,
    MultiStepLRLinearWarmUp,
    ReduceLROnPlateauWithWarmup,
    SparsifiedSGDLR,
    SQRTTimeDecayLR,
    SQRTTimeDecayLRWithWarmup,
    TimeDecayLR,
)
