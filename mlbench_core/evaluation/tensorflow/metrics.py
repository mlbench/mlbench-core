r"""Define tensorflow metrics."""

import tensorflow as tf


class TopKAccuracy(object):
    """Compute the top-k accuracy of logits.

    Args:
        logits (:obj:`tf.Tensor`): input tensor
        labels (:obj:`tf.Tensor`): input one-hot encoded tensor.
        topkk (:obj:`int`, optional): Defaults to 1. top k accuracy.
    """

    def __init__(self, logits, labels, topk=1):
        labels = tf.cast(labels, tf.int32)
        true_classes = tf.argmax(labels, axis=1)

        # predicted classes
        pred_probs = tf.nn.softmax(logits, name="softmax_tensor")
        pred_classes = tf.argmax(pred_probs, axis=1)

        # get metrics.
        with tf.name_scope("metrics"):
            if topk == 1:
                self.name = "Prec@1"
                self.metric_op = (
                    tf.reduce_mean(
                        tf.cast(tf.equal(true_classes, pred_classes), tf.float32)
                    )
                    * 100.0
                )
            else:
                topk_op = tf.nn.in_top_k(
                    predictions=pred_probs, targets=true_classes, k=topk
                )
                self.name = "Prec@" + str(topk)
                self.metric_op = tf.reduce_mean(tf.cast(topk_op, tf.float32)) * 100.0


def topk_accuracy_with_logits(logits, labels, k=1):
    """Compute the top-k accuracy of logits.

    Args:
        logits (:obj:`tf.Tensor`): input tensor
        labels (:obj:`tf.Tensor`): input one-hot encoded tensor.
        k (:obj:`int`, optional): Defaults to 1. top k accuracy.

    Returns:
        :obj:`tf.Tensor`: a scalar tensor of the accuracy (between 0 and 1).
    """
    m = TopKAccuracy(logits=logits, labels=labels, topk=k)
    return {"name": m.name, "value": m.metric_op}
