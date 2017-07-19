# coding = utf-8

import tensorflow as tf

with tf.variable_scope('root') as scope:
    var1 = tf.get_variable('var1', shape=[1], dtype=tf.float32)
    var2 = tf.get_variable('var2', shape=[1], dtype=tf.float32)
    sum1 = var1 + var2
    with tf.name_scope('plus'):
        sum2 = var1 + var2
        var3 = tf.get_variable('var3', shape=[1], dtype=tf.float32)

print(var1.name)
print(var2.name)

# 观察sum1,sum2,var3之间的区别
print(sum1.name)
print(sum2.name)
print(var3.name)


