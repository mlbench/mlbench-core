mlbench_core.optim
------------------

.. autoapimodule:: mlbench_core.optim
.. currentmodule:: mlbench_core.optim


pytorch
~~~~~~~
.. autoapimodule:: mlbench_core.optim.pytorch
.. currentmodule:: mlbench_core.optim.pytorch


Optimizers
++++++++++

The optimizers in this module are not distributed. Their purpose is to implement logic that
can be inherited by distributed optimizers.

.. autoapimodule:: mlbench_core.optim.pytorch.optim
.. currentmodule:: mlbench_core.optim.pytorch.optim


SparsifiedSGD
'''''''''''''

.. autoapiclass:: SparsifiedSGD
    :members:

SignSGD
'''''''''''''

.. autoapiclass:: SignSGD
    :members:

Centralized (Synchronous) Optimizers
++++++++++++++++++++++++++++++++++++

The optimizers in this module are all distributed and synchronous: workers advance in a synchronous manner. All workers
communicate with each other using `all_reduce` or `all_gather` operations.

.. autoapimodule:: mlbench_core.optim.pytorch.centralized
.. currentmodule:: mlbench_core.optim.pytorch.centralized

Generic Centralized Optimizer
+++++++++++++++++++++++++++++

.. autoapiclass:: GenericCentralizedOptimizer
    :members:

CentralizedSGD
''''''''''''''

.. autoapiclass:: CentralizedSGD
    :show-inheritance:
    :members:

CentralizedAdam
'''''''''''''''

.. autoapiclass:: CentralizedAdam
    :show-inheritance:
    :members:

CustomCentralizedOptimizer
''''''''''''''''''''''''''

.. autoapiclass:: CustomCentralizedOptimizer
    :show-inheritance:
    :members:

CentralizedSparsifiedSGD
''''''''''''''''''''''''

.. autoapiclass:: CentralizedSparsifiedSGD
    :members:

PowerSGD
''''''''

.. autoapiclass:: PowerSGD
    :members:

Decentralized (Asynchronous) Optimizers
+++++++++++++++++++++++++++++++++++++++

The optimizers in this module are all distributed and asynchronous: workers advance independently from each other,
and communication patterns follow an arbitrary graph.

.. autoapimodule:: mlbench_core.optim.pytorch.decentralized
.. currentmodule:: mlbench_core.optim.pytorch.decentralized

DecentralizedSGD
''''''''''''''''

.. autoapiclass:: DecentralizedSGD
    :members:


.. rubric:: References

.. bibliography:: optim.bib
   :cited:

Mixed Precision Optimizers
++++++++++++++++++++++++++

.. autoapimodule:: mlbench_core.optim.pytorch.fp_optimizers
.. currentmodule:: mlbench_core.optim.pytorch.fp_optimizers

FP16Optimizer
'''''''''''''

.. autoapiclass:: FP16Optimizer
    :members:
