r"""Define loss functions."""

import tensorflow as tf


def softmax_cross_entropy_with_logits_v2_l2_regularized(
    logits, labels, l2, loss_filter_fn
):
    """Return an op for computing cross entropy with weight decay.

    The `labels` are assumed to be one-hot encoded. The loss filter function excludes some
    tensors from computing weight decay.

    Args:
        logits (:obj:`tf.Tensor`): input logits tensor.
        labels (:obj:`tf.Tensor`): input one-hot encoded tensor.
        l2 (:obj:`float`): size of weight decay
        loss_filter_fn (:obj:`callable`): filter function.

    Returns:
        :obj:`tf.Tensor`: a scalar tensor
    """
    labels = tf.cast(labels, tf.int32)
    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        )

        loss = cross_entropy + l2 * tf.add_n(
            [
                tf.nn.l2_loss(v)
                for v in tf.trainable_variables()
                if loss_filter_fn(v.name)
            ]
        )
    return loss
