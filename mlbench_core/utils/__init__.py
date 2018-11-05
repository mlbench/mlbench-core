import argparse


class Tracker(argparse.Namespace):
    """A class to track running stats."""
    pass


try:
    import torch
    from . import pytorch
except ImportError:
    pass
