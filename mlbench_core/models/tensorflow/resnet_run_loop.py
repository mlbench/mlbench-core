"""
Adapted from https://github.com/tensorflow/models/blob/master/official/resnet/resnet_run_loop.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import multiprocessing
import os

from absl import flags
import tensorflow as tf
from tensorflow.contrib.data.python.ops import threadpool

from . import resnet_model

from mlbench_core.utils.tensorflow import distribution_utils


def resnet_model_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, resnet_version, loss_scale,
                    loss_filter_fn=None, dtype=resnet_model.DEFAULT_DTYPE,
                    fine_tune=False):
    """Shared functionality for different resnet model_fns.
    Initializes the ResnetModel representing the model layers
    and uses that model to build the necessary EstimatorSpecs for
    the `mode` in question. For training, this means building losses,
    the optimizer, and the train op that get passed into the EstimatorSpec.
    For evaluation and prediction, the EstimatorSpec is returned without
    a train op, but with the necessary parameters for the given mode.
    Args:
        features: tensor representing input images
        labels: tensor representing class labels for all input images
        mode: current estimator mode; should be one of
        `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
        model_class: a class representing a TensorFlow model that has a __call__
        function. We assume here that this is a subclass of ResnetModel.
        resnet_size: A single integer for the size of the ResNet model.
        weight_decay: weight decay loss rate used to regularize learned variables.
        learning_rate_fn: function that returns the current learning rate given
        the current global_step
        momentum: momentum term used for optimization
        data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
        resnet_version: Integer representing which version of the ResNet network to
        use. See README for details. Valid values: [1, 2]
        loss_scale: The factor to scale the loss for numerical stability. A detailed
        summary is present in the arg parser help text.
        loss_filter_fn: function that takes a string variable name and returns
        True if the var should be included in loss calculation, and False
        otherwise. If None, batch_normalization variables will be excluded
        from the loss.
        dtype: the TensorFlow dtype to use for calculations.
        fine_tune: If True only train the dense layers(final layers).
    Returns:
        EstimatorSpec parameterized according to the input params and the
        current mode.
    """

    # Generate a summary node for the images
    tf.summary.image('images', features, max_outputs=6)
    # Checks that features/images have same data type being used for calculations.
    assert features.dtype == dtype

    model = model_class(resnet_size, data_format, resnet_version=resnet_version,
                        dtype=dtype)

    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    # This acts as a no-op if the logits are already in fp32 (provided logits are
    # not a SparseTensor). If dtype is is low precision, logits must be cast to
    # fp32 for numerical stability.
    logits = tf.cast(logits, tf.float32)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name
    loss_filter_fn = loss_filter_fn or exclude_batch_norm

    # Add weight decay to the loss.
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
         if loss_filter_fn(v.name)])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum
        )

        def _dense_grad_filter(gvs):
            """Only apply gradient updates to the final layer.
            This function is used for fine tuning.
            Args:
                gvs: list of tuples with gradients and variable info
            Returns:
                filtered gradients so that only the dense layer remains
            """
            return [(g, v) for g, v in gvs if 'dense' in v.name]

        if loss_scale != 1:
            # When computing fp16 gradients, often intermediate tensor values are
            # so small, they underflow to 0. To avoid this, we multiply the loss by
            # loss_scale to make these tensor values loss_scale times bigger.
            scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

            if fine_tune:
                scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

            # Once the gradient computation is complete we can scale the gradients
            # back to the correct scale before passing them to the optimizer.
            unscaled_grad_vars = [(grad / loss_scale, var)
                                  for grad, var in scaled_grad_vars]
            minimize_op = optimizer.apply_gradients(
                unscaled_grad_vars, global_step)
        else:
            grad_vars = optimizer.compute_gradients(loss)

            if fine_tune:
                grad_vars = _dense_grad_filter(grad_vars)
            minimize_op = optimizer.apply_gradients(grad_vars, global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits,
                                                    targets=labels,
                                                    k=5,
                                                    name='top_5_op'))
    metrics = {'accuracy': accuracy,
               'accuracy_top_5': accuracy_top_5}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def override_flags_and_set_envars_for_gpu_thread_pool(config):
    """Override flags and set env_vars for performance.
    These settings exist to test the difference between using stock settings
    and manual tuning. It also shows some of the ENV_VARS that can be tweaked to
    squeeze a few extra examples per second.  These settings are defaulted to the
    current platform of interest, which changes over time.
    On systems with small numbers of cpu cores, e.g. under 8 logical cores,
    setting up a gpu thread pool with `tf_gpu_thread_mode=gpu_private` may perform
    poorly.
    Args:
        config: Current flags, which will be adjusted possibly overriding
        what has been set by the user on the command-line.
    """
    cpu_count = multiprocessing.cpu_count()
    tf.logging.info('Logical CPU cores: %s', cpu_count)

    # Sets up thread pool for each GPU for op scheduling.
    per_gpu_thread_count = 1
    total_gpu_thread_count = per_gpu_thread_count * config.world_size
    os.environ['TF_GPU_THREAD_MODE'] = config.tf_gpu_thread_mode
    os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)

    # TODO: logging with `tf.logging` or native logger?
    tf.logging.info('TF_GPU_THREAD_COUNT: %s',
                    os.environ['TF_GPU_THREAD_COUNT'])
    tf.logging.info('TF_GPU_THREAD_MODE: %s', os.environ['TF_GPU_THREAD_MODE'])

    # Reduces general thread pool by number of threads used for GPU pool.
    main_thread_count = cpu_count - total_gpu_thread_count
    config.inter_op_parallelism_threads = main_thread_count

    # Sets thread count for tf.data. Logical cores minus threads assign to the
    # private GPU pool along with 2 thread per GPU for event monitoring and
    # sending / receiving tensors.
    num_monitoring_threads = 2 * config.world_size
    config.datasets_num_private_threads = (cpu_count - total_gpu_thread_count
                                           - num_monitoring_threads)


def resnet_main(config, model_function, input_function, dataset_name, shape=None):
    """Shared main loop for ResNet Models.

    TODO: Put the resnet main to controlflow.

    Args:
    config: An object containing parsed flags. See define_resnet_flags()
        for details.
    model_function: the function that instantiates the Model and builds the
        ops for train/eval. This will be passed directly into the estimator.
    input_function: the function that processes the dataset and returns a
        dataset that the estimator can train on. This will be wrapped with
        all the relevant flags for running and passed to estimator.
    dataset_name: the name of the dataset for training and evaluation. This is
        used for logging purpose.
    shape: list of ints representing the shape of the images used for training.
        This is only used if config.export_dir is passed.
    """

    # Ensures flag override logic is only executed if explicitly triggered.
    if config.tf_gpu_thread_mode:
        override_flags_and_set_envars_for_gpu_thread_pool(config)

    # Creates session config. allow_soft_placement = True, is required for
    # multi-GPU and is not harmful for other modes.
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=config.inter_op_parallelism_threads,
        intra_op_parallelism_threads=config.intra_op_parallelism_threads,
        allow_soft_placement=True)

    # all_reduce_alg
    distribution_strategy = distribution_utils.get_distribution_strategy(
        config.world_size, config.all_reduce_alg)

    # Creates a `RunConfig` that checkpoints every 24 hours which essentially
    # results in checkpoints determined only by `epochs_between_evals`.
    run_config = tf.estimator.RunConfig(
        # TODO:
        # train_distribute=distribution_strategy,
        session_config=session_config,
        save_checkpoints_secs=60 * 60 * 24)

    # Initializes model with all but the dense layer from pretrained ResNet.
    if (hasattr(config, "pretrained_model_checkpoint_path") and
            config.pretrained_model_checkpoint_path is not None):
        # TODO: should we include pretrained model for evaluation?
        # The pretrained model can be trained on validation set.
        warm_start_settings = tf.estimator.WarmStartSettings(
            config.pretrained_model_checkpoint_path,
            vars_to_warm_start='^(?!.*dense)')
    else:
        warm_start_settings = None

    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=config.model_dir, config=run_config,
        warm_start_from=warm_start_settings, params={
            'resnet_size': int(config.resnet_size),
            'data_format': config.data_format,
            'batch_size': config.batch_size,
            'resnet_version': int(config.resnet_version),
            # official/utils/flags/_performance.py
            'loss_scale': config.tf_loss_scale,
            'dtype': config.tf_dtype,
            'fine_tune': config.fine_tune
        })

    run_params = {
        'batch_size': config.batch_size,
        'dtype': config.tf_dtype,
        'resnet_size': config.resnet_size,
        'resnet_version': config.resnet_version,
        'synthetic_data': config.use_synthetic_data,
        'train_epochs': config.train_epochs,
    }

    # TODO: add back logger
    # benchmark_logger = logger.get_benchmark_logger()
    # benchmark_logger.log_run_info('resnet', dataset_name, run_params,
    #                               test_id=config.benchmark_test_id)

    # TODO: add hooks for training
    train_hooks = []
    # train_hooks = hooks_helper.get_train_hooks(
    #     config.hooks,
    #     model_dir=config.model_dir,
    #     batch_size=config.batch_size)

    def input_fn_train(num_epochs):
        return input_function(
            is_training=True,
            data_dir=config.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                config.batch_size, config.world_size),
            num_epochs=num_epochs,
            dtype=config.tf_dtype,
            datasets_num_private_threads=config.datasets_num_private_threads,
            num_parallel_batches=config.datasets_num_parallel_batches)

    def input_fn_eval():
        return input_function(
            is_training=False,
            data_dir=config.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                config.batch_size, config.world_size),
            num_epochs=1,
            dtype=config.tf_dtype)

    if config.eval_only or not config.train_epochs:
        # If --eval_only is set, perform a single loop with zero train epochs.
        schedule, n_loops = [0], 1
    else:
        # Compute the number of times to loop while training. All but the last
        # pass will train for `epochs_between_evals` epochs, while the last will
        # train for the number needed to reach `training_epochs`. For instance if
        #   train_epochs = 25 and epochs_between_evals = 10
        # schedule will be set to [10, 10, 5]. That is to say, the loop will:
        #   Train for 10 epochs and then evaluate.
        #   Train for another 10 epochs and then evaluate.
        #   Train for a final 5 epochs (to reach 25 epochs) and then evaluate.
        n_loops = math.ceil(config.train_epochs /
                            config.epochs_between_evals)
        schedule = [
            config.epochs_between_evals for _ in range(int(n_loops))]
        # over counting.
        schedule[-1] = config.train_epochs - sum(schedule[:-1])

    for cycle_index, num_train_epochs in enumerate(schedule):
        tf.logging.info('Starting cycle: %d/%d', cycle_index, int(n_loops))

        if num_train_epochs:
            print("classifier.train")
            classifier.train(input_fn=lambda: input_fn_train(num_train_epochs),
                             hooks=train_hooks, max_steps=config.max_train_steps)

        tf.logging.info('Starting to evaluate.')

        # config.max_train_steps is generally associated with testing and
        # profiling. As a result it is frequently called with synthetic data, which
        # will iterate forever. Passing steps=config.max_train_steps allows the
        # eval (which is generally unimportant in those circumstances) to terminate.
        # Note that eval will run for max_train_steps each loop, regardless of the
        # global_step count.
        eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                           steps=config.max_train_steps)

        benchmark_logger.log_evaluation_result(eval_results)

        if model_helpers.past_stop_threshold(
                config.stop_threshold, eval_results['accuracy']):
            break

    if config.export_dir is not None:
        # Exports a saved model for the given classifier.
        export_dtype = config.dtype
        if config.image_bytes_as_serving_input:
            input_receiver_fn = functools.partial(
                image_bytes_serving_input_fn, shape, dtype=export_dtype)
        else:
            input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
                shape, batch_size=config.batch_size, dtype=export_dtype)
        classifier.export_savedmodel(config.export_dir, input_receiver_fn,
                                     strip_default_attrs=True)
