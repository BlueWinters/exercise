# coding=utf-8
# date: 2018-4-12,21:32:45
# name: smz


import tensorflow as tf


def ex_metrics_mean():
    def get_mean(values):
        with tf.name_scope('mean'):
            mean, update_op = tf.metrics.mean(values)  # values = [1, 2, 3, 4, 5]
            with tf.control_dependencies([update_op]):
                mean = tf.identity(mean)
            return mean

    g = tf.Graph()
    g.as_default()
    values = tf.convert_to_tensor([1, 2, 3, 4, 5], dtype=tf.float32)
    mean= get_mean(values)
    init_global_variables = tf.global_variables_initializer()
    init_local_variables = tf.local_variables_initializer()
    g.finalize()

    sess = tf.InteractiveSession(graph=tf.get_default_graph())
    sess.run(init_global_variables)
    sess.run(init_local_variables)
    print sess.run(values)
    for i in range(5):
        print 'mean:', sess.run(mean)

# out:
# [1. 2. 3. 4. 5.]
# mean: 0.0
# mean: 6.0
# mean: 4.5
# mean: 4.0
# mean: 3.75

# 不符合预期
# 预期值为：
# [1. 2. 3. 4. 5.]
# mean: 3.0
# mean: 3.0
# mean: 3.0
# mean: 3.0
# mean: 3.0

# 理由：
# iter: 1
# total = 1+2+3+4+5 = 15
# count = 1+1+1+1+1 = 5
# mean = 15 / 5 = 3
# iter: 2
# total = 15 + 15 = 30
# count = 5 + 5 = 10
# mean = 30 / 10 = 3
# 类推
# 出错原因： 不明


if __name__ == '__main__':
   ex_metrics_mean()
