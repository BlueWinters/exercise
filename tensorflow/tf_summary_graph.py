# coding=utf-8

import tensorflow as tf

def set_fc_vars(in_dim, out_dim):
    W = tf.get_variable(name='W', shape=[in_dim, out_dim], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable(name='b', shape=[1, out_dim], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1))

def calc_fc(input, name='fc'):
    with tf.name_scope(name) as scope:
        W = tf.get_variable(name='W')
        b = tf.get_variable(name='b')
        a = tf.matmul(input, W) + b
    return a

def combine_calc_fc(input, in_dim, out_dim, name='fc'):
    with tf.name_scope('fc'):
        W = tf.get_variable(name='W', shape=[in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable(name='b', shape=[1, out_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        a = tf.matmul(input, W) + b
    return a

def summary_graph1():
    input_size = 25
    batch_size = 784
    network1 = [input_size, 36, 100]
    network2 = [input_size, 49, 100]
    network_12 = [100, 100]

    input1 = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_size], name='input1')
    input2 = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_size], name='input2')

    with tf.variable_scope('network1'):
        with tf.variable_scope('layer1') as scope1:
            set_fc_vars(network1[0], network1[1])
        with tf.variable_scope(scope1) as scope1_reuse:
            scope1_reuse.reuse_variables()
            a1 = calc_fc(input1)

        with tf.variable_scope('layer2') as scope2:
            set_fc_vars(network1[1], network1[2])
        with tf.variable_scope(scope2) as scope2_reuse:
            scope2_reuse.reuse_variables()
            a1 = calc_fc(a1)


    with tf.variable_scope('network2'):
        with tf.variable_scope('layer1') as scope1:
            set_fc_vars(network2[0], network2[1])
        with tf.variable_scope(scope1, reuse=True) as scope1_reuse:
            a2 = calc_fc(input2)

        with tf.variable_scope('layer2') as scope2:
            set_fc_vars(network2[1], network2[2])
        with tf.variable_scope(scope2, reuse=True) as scope2_reuse:
            a2 = calc_fc(a2)

    with tf.name_scope('concat') as scope_concat:
        a = tf.concat([a1, a2], axis=0)

    with tf.variable_scope('network_12'):
        with tf.variable_scope('layer3') as scope_layer3:
            set_fc_vars(network_12[0], network2[1])
        with tf.variable_scope(scope_layer3, reuse=True) as scope_layer3_reuse:
            aa = calc_fc(a)

    sess = tf.Session()
    merged = tf.summary.merge_all()
    tf.summary.FileWriter('./summary', sess.graph)
    sess.close()

    # 问题：layer中的变量会单独来自一个数据流

def summary_graph2():
    input_size = 25
    batch_size = 784
    network = [input_size, 36, 100]

    input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_size], name='input')

    with tf.variable_scope('layer1') as scope:
        a = combine_calc_fc(input, network[0], network[1])
    with tf.variable_scope('layer2') as scope:
        a = combine_calc_fc(a, network[1], network[2])

    sess = tf.Session()
    merged = tf.summary.merge_all()
    tf.summary.FileWriter('./summary', sess.graph)
    sess.close()

    # 2017.07.29
    # 这是目前建议的方式，因为在tensorboard中能够得到更好的graph visual


if __name__ == '__main__':
    # summary_graph()
    # summary_graph2()
    pass