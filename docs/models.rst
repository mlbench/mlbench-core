mlbench_core.models
-------------------
.. autoapimodule:: mlbench_core.models
.. currentmodule:: mlbench_core.models

pytorch
~~~~~~~

Since `Kuang Liu<https://github.com/kuangliu/pytorch-cifar>` has already included many classical
neural network models. We use their implementation direclty for 

- VGG

.. autoapimodule:: mlbench_core.models.pytorch
.. currentmodule:: mlbench_core.models.pytorch


linear_models
+++++++++++++

.. autoapimodule:: mlbench_core.models.pytorch.linear_models
.. currentmodule:: mlbench_core.models.pytorch.linear_models


LogisticRegression
''''''''''''''''''

.. autoapiclass:: LogisticRegression
    :members:

LinearRegression
''''''''''''''''''

.. autoapiclass:: LinearRegression
    :members:


resnet
++++++
.. autoapimodule:: mlbench_core.models.pytorch.resnet
.. currentmodule:: mlbench_core.models.pytorch.resnet

ResNetCIFAR
'''''''''''

.. autoapiclass:: ResNetCIFAR
    :members:


ResNet18_CIFAR10
''''''''''''''''

.. autoapiclass:: ResNetCIFAR
    :members:


.. rubric:: References

.. bibliography:: models.bib
   :cited:


tensorflow
~~~~~~~~~~

.. autoapimodule:: mlbench_core.models.tensorflow
.. currentmodule:: mlbench_core.models.tensorflow

resnet
++++++

.. autoapimodule:: mlbench_core.models.tensorflow.resnet_model
.. currentmodule:: mlbench_core.models.tensorflow.resnet_model


.. autoapifunction:: fixed_padding

.. autoapifunction:: conv2d_fixed_padding

.. autoapifunction:: block_layer

.. autoapifunction:: batch_norm


Model
'''''

.. autoapiclass:: Model
    :members:


Cifar10Model
''''''''''''

.. autoapiclass:: Cifar10Model
    :members:


