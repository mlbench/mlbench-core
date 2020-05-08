mlbench_core.controlflow
------------------------

.. autoapimodule:: mlbench_core.controlflow
.. currentmodule:: mlbench_core.controlflow

pytorch
~~~~~~~

.. autoapimodule:: mlbench_core.controlflow.pytorch
.. currentmodule:: mlbench_core.controlflow.pytorch

Controlflow
+++++++++++

.. autoapifunction:: validation_round

.. autoapifunction:: record_train_batch_stats

.. autoapifunction:: record_validation_stats

CheckpointsEvaluationControlFlow
++++++++++++++++++++++++++++++++

.. autoapiclass:: CheckpointsEvaluationControlFlow
    :members:

TrainValidation (Deprecated)
++++++++++++++++++++++++++++

.. autoapiclass:: mlbench_core.controlflow.pytorch.train_validation.TrainValidation
    :members:

    .. autoapimethod:: __call__

Helpers
+++++++

.. autoapimodule:: mlbench_core.controlflow.pytorch.helpers
.. currentmodule:: mlbench_core.controlflow.pytorch.helpers

.. autoapifunction:: maybe_range
.. autoapifunction:: convert_dtype
.. autoapifunction:: prepare_batch
.. autoapifunction:: iterate_dataloader



tensorflow
~~~~~~~~~~

.. autoapimodule:: mlbench_core.controlflow.tensorflow
.. currentmodule:: mlbench_core.controlflow.tensorflow


TrainValidation
+++++++++++++++

.. autoapiclass:: TrainValidation
    :members:

    .. autoapimethod:: __call__
