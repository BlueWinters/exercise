# coding=utf-8

import tensorflow as tf


def variable_dropout():
    sess = tf.InteractiveSession()
    var = tf.get_variable('var', shape=[4,5], initializer=tf.constant_initializer(1))
    var_dropout = tf.nn.dropout(var, 0.25)

    sess.run(tf.global_variables_initializer())
    print(var.eval())
    print(var_dropout.eval())
    print(var_dropout.eval())
    sess.close()

    # 输出
    # 每次结果都不一样


if __name__ == '__main__':
    variable_dropout()