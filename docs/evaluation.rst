mlbench_core.evaluation
-----------------------
.. autoapimodule:: mlbench_core.evaluation
.. currentmodule:: mlbench_core.evaluation

pytorch
~~~~~~~

.. autoapimodule:: mlbench_core.evaluation.pytorch
.. currentmodule:: mlbench_core.evaluation.pytorch

criterion
+++++++++

.. autoapimodule:: mlbench_core.evaluation.pytorch.criterion
.. currentmodule:: mlbench_core.evaluation.pytorch.criterion


BCELossRegularized
''''''''''''''''''

.. autoapiclass:: BCELossRegularized
    :members:


MSELossRegularized
''''''''''''''''''

.. autoapiclass:: MSELossRegularized
    :members:

.. autoapiclass:: LabelSmoothing
    :members:

metrics
+++++++

.. autoapimodule:: mlbench_core.evaluation.pytorch.metrics
.. currentmodule:: mlbench_core.evaluation.pytorch.metrics


TopKAccuracy
''''''''''''

.. autoapiclass:: TopKAccuracy
    :members:

    .. autoapimethod:: __call__

tensorflow
~~~~~~~~~~

criterion
+++++++++

.. autoapimodule:: mlbench_core.evaluation.tensorflow.criterion
.. currentmodule:: mlbench_core.evaluation.tensorflow.criterion


softmax_cross_entropy_with_logits_v2_l2_regularized
'''''''''''''''''''''''''''''''''''''''''''''''''''

.. autoapifunction:: softmax_cross_entropy_with_logits_v2_l2_regularized

metrics
+++++++

.. autoapimodule:: mlbench_core.evaluation.tensorflow.metrics
.. currentmodule:: mlbench_core.evaluation.tensorflow.metrics

topk_accuracy
'''''''''''''

.. autoapifunction:: topk_accuracy_with_logits
