mlbench_core.evaluation
-----------------------
.. automodule:: mlbench_core.evaluation
.. currentmodule:: mlbench_core.evaluation

pytorch
~~~~~~~

.. automodule:: mlbench_core.evaluation.pytorch
.. currentmodule:: mlbench_core.evaluation.pytorch

criterion
+++++++++

.. automodule:: mlbench_core.evaluation.pytorch.criterion
.. currentmodule:: mlbench_core.evaluation.pytorch.criterion


BCELossRegularized
''''''''''''''''''

.. autoclass:: BCELossRegularized
    :members:


MSELossRegularized
''''''''''''''''''

.. autoclass:: MSELossRegularized
    :members:


metrics
+++++++

.. automodule:: mlbench_core.evaluation.pytorch.metrics
.. currentmodule:: mlbench_core.evaluation.pytorch.metrics


TopKAccuracy
''''''''''''

.. autoclass:: TopKAccuracy
    :members:

    .. automethod:: __call__


tensorflow
~~~~~~~~~~

criterion
+++++++++

.. automodule:: mlbench_core.evaluation.tensorflow.criterion
.. currentmodule:: mlbench_core.evaluation.tensorflow.criterion


softmax_cross_entropy_with_logits_v2_l2_regularized
'''''''''''''''''''''''''''''''''''''''''''''''''''

.. autofunction:: softmax_cross_entropy_with_logits_v2_l2_regularized

metrics
+++++++

.. automodule:: mlbench_core.evaluation.tensorflow.metrics
.. currentmodule:: mlbench_core.evaluation.tensorflow.metrics

topk_accuracy
'''''''''''''

.. autofunction:: topk_accuracy_with_logits

