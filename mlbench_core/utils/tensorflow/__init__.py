"""Initialize environment for pytorch."""

import tensorflow as tf


def _init_cleanup(config):
    r"""Cleanup legacy files like logs, output."""
    print("=> Initial cleanup")


def _init_log(config):
    print("=> Initialize log")


def _init_tensorflow(config):
    print("=> Initialize TensorFlow")


def initialize_backends(config):
    """Initializes the backends.

    Sets up logging, sets up tensorflow and configures paths
    correctly.

    Args:
        config (:obj:`types.SimpleNamespace`): a global object containing all of the config.

    Returns:
        (:obj:`types.SimpleNamespace`): a global object containing all of the config.
    """
    _init_cleanup(config)

    _init_log(config)

    _init_tensorflow(config)
    return config


def default_session_config(tf_allow_soft_placement, tf_log_device_placement, tf_gpu_mem):
    """Initialize session configuration."""
    session_conf = tf.ConfigProto(
        allow_soft_placement=tf_allow_soft_placement,
        log_device_placement=tf_log_device_placement)

    session_conf.gpu_options.allow_growth = False  # True
    session_conf.gpu_options.per_process_gpu_memory_fraction = tf_gpu_mem
    return session_conf
