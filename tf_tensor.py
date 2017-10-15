# coding=utf-8

import tensorflow as tf


def tensor_start_tensor():
    t1 = tf.get_variable(name='t1',shape=[2,5], dtype=tf.float32)
    t2 = tf.get_variable(name='t2',shape=[5,3], dtype=tf.float32)
    t3 = tf.matmul(t1, t2)
    t4 = t3 * 10

    print(t3.get_shape().as_list())
    print(t4.get_shape().as_list())








if __name__ == '__main__':
    tensor_start_tensor()
    pass
