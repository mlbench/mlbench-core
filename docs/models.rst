mlbench_core.models
-------------------
.. automodule:: mlbench_core.models
.. currentmodule:: mlbench_core.models

pytorch
~~~~~~~

Since `Kuang Liu<https://github.com/kuangliu/pytorch-cifar>` has already included many classical
neural network models. We use their implementation direclty for 

- VGG

.. automodule:: mlbench_core.models.pytorch
.. currentmodule:: mlbench_core.models.pytorch


linear_models
+++++++++++++

.. automodule:: mlbench_core.models.pytorch.linear_models
.. currentmodule:: mlbench_core.models.pytorch.linear_models


LogisticRegression
''''''''''''''''''

.. autoclass:: LogisticRegression
    :members:

LinearRegression
''''''''''''''''''

.. autoclass:: LinearRegression
    :members:


resnet
++++++
.. automodule:: mlbench_core.models.pytorch.resnet
.. currentmodule:: mlbench_core.models.pytorch.resnet

ResNetCIFAR
'''''''''''

.. autoclass:: ResNetCIFAR
    :members:


ResNet18_CIFAR10
''''''''''''''''

.. autoclass:: ResNetCIFAR
    :members:


.. rubric:: References

.. bibliography:: models.bib
   :cited:


tensorflow
~~~~~~~~~~

.. automodule:: mlbench_core.models.tensorflow
.. currentmodule:: mlbench_core.models.tensorflow

resnet
++++++

.. automodule:: mlbench_core.models.tensorflow.resnet_model
.. currentmodule:: mlbench_core.models.tensorflow.resnet_model


.. autofunction:: fixed_padding
.. autofunction:: conv2d_fixed_padding 
.. autofunction:: block_layer
.. autofunction:: batch_norm

Model
'''''

.. autoclass:: Model
    :members:


Cifar10Model
''''''''''''

.. autoclass:: Cifar10Model
    :members:


