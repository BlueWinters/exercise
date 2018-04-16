# coding=utf-8

import tensorflow as tf


def build_top_k(predictions, labels, k, name):
    with tf.name_scope(name):
        target = tf.cast(tf.argmax(labels, axis=1), tf.int32)
        in_top_k = tf.to_float(tf.nn.in_top_k(predictions, target, k=k))
        return tf.reduce_sum(tf.cast(in_top_k, tf.float32))

    # top k的正确率
    # tf.nn.in_top_k的输入必须是sparse的形式（相对one_hot的那种）


if __name__ == '__main__':
    pass