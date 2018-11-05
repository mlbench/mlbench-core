r"""Define tensorflow metrics."""

import tensorflow as tf


def topk_accuracy_with_logits(logits, labels, k=1):
    """Compute the top-k accuracy of logits.

    Args:
        logits (:obj:`tf.Tensor`): input tensor
        labels (:obj:`tf.Tensor`): input one-hot encoded tensor.
        k (:obj:`int`, optional): Defaults to 1. top k accuracy.

    Returns:
        :obj:`tf.Tensor`: a scalar tensor of the accuracy (between 0 and 1).
    """

    labels = tf.cast(labels, tf.int32)
    true_classes = tf.argmax(labels, axis=1)

    # predicted classes
    pred_probs = tf.nn.softmax(logits, name='softmax_tensor')
    pred_classes = tf.argmax(pred_probs, axis=1)

    # get metrics.
    with tf.name_scope("metrics"):
        if k == 1:
            return {"name": "top1", "value": tf.reduce_mean(
                tf.cast(tf.equal(true_classes, pred_classes), tf.float32))}
        else:
            topk = tf.nn.in_top_k(predictions=pred_probs,
                                  targets=true_classes, k=k)
            return {"name": "top" + str(k), "value": tf.reduce_mean(tf.cast(topk, tf.float32))}
