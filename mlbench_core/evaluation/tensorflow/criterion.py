r"""Define loss functions and metrics."""

import tensorflow as tf


def get_outputs(logits, labels, l2, loss_filter_fn):
    labels = tf.cast(labels, tf.int32)
    true_classes = tf.argmax(labels, axis=1)
    pred_probs = tf.nn.softmax(logits, name='softmax_tensor')
    pred_classes = tf.argmax(pred_probs, axis=1)

    with tf.name_scope("loss"):
        # TODO: Do we need to apply soft max again?
        # TODO: Add names for both operators?
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=labels))
        loss = cross_entropy + l2 * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()
             if loss_filter_fn(v.name)])

    # get metrics.
    with tf.name_scope("metrics"):
        prec1 = tf.reduce_mean(
            tf.cast(tf.equal(true_classes, pred_classes), tf.float32)
        )

        prec5 = tf.reduce_mean(
            tf.cast(
                tf.nn.in_top_k(
                    predictions=pred_probs,
                    targets=true_classes,
                    k=5),
                tf.float32)
        )

    # TODO: Change the metrics to be customized.
    return loss, [prec1, prec5]
