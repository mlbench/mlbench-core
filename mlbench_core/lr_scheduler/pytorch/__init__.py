"""Scheduling Learning Rates.

.. rubric:: References

.. [ginsburg2018large] Ginsburg, Boris and Gitman, Igor and You, Yang
    Large Batch Training of Convolutional Networks with Layer-wise Adaptive Rate Scaling

.. [leslie2017cyclical] Leslie N. Smith
    Cyclical Learning Rates for Training Neural Networks

.. [goyal2017accurate] Goyal, Priya, et al.
    Accurate, large minibatch SGD: training imagenet in 1 hour.

.. [smith2017super] Smith, Leslie N., and Nicholay Topin.
    Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates.


"""

from .lr import triangular_learning_rates, cyclical_learning_rates, \
    multistep_learning_rates_with_warmup
