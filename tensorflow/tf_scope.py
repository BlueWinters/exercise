# coding=utf-8

import tensorflow as tf

def name_scope_and_variable_scope():
    # name_scope和variable_scope的区别
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

    # 输出
    # root/var1:0
    # root/var2:0
    # root/add:0
    # root/plus/add:0
    # root/var3:0

def variable_scope_reuse():
    # 变量的创建，var1和var2
    with tf.variable_scope('root') as scope:
        var1 = tf.get_variable('var1', shape=[1], dtype=tf.float32)
        print(var1.name)
    with tf.variable_scope('root') as scope:
        var2 = tf.get_variable('var2', shape=[1], dtype=tf.float32)
        print(var2.name)

    # 变量的reuse，var1和var1_2的name是一样的
    with tf.variable_scope('root', reuse=True) as scope:
        # scope.reuse_variables()
        var1_2 = tf.get_variable('var1', shape=[1], dtype=tf.float32)
        print(var1_2.name)

    # 输出
    # root/var1:0
    # root/var2:0
    # root/var1:0

def variable_scope_init_reuse():
    with tf.variable_scope('root', reuse=True) as scope:
        var1 = tf.get_variable('var1', shape=[1], dtype=tf.float32)
        print(var1.name)

    # 错误输出
    # Variable root/var1 does not exist, or was not created with tf.get_variable().
    # Did you mean to set reuse=None in VarScope?

def variable_scope_reuse_False():
    with tf.variable_scope('root', reuse=False) as scope:
        var1 = tf.get_variable('var1', shape=[1], dtype=tf.float32)
    print(var1.name)

    with tf.variable_scope('root', reuse=None) as scope:
        var2 = tf.get_variable('var2', shape=[1], dtype=tf.float32)
    print(var2.name)

    # 输出
    # root/var1:0
    # root/var2:0

def name_scope_reuse():
    with tf.variable_scope('root'):
        var1 = tf.get_variable('var1', shape=[1])
        var2 = tf.get_variable('var2', shape=[1])

        with tf.name_scope('sub1') as scope:
            sub1 = var1 + var2
            print(sub1.name)
        with tf.name_scope('sub1'):
            sub2 = var1 + var2
            print(sub2.name)

    # 输出
    # root/sub1/add:0
    # root/sub1_1/add:0

def sub_variable_scope_reuse():
    with tf.variable_scope('root'):
        with tf.variable_scope('sub1'):
            var1 = tf.get_variable('var1', shape=[1])
        with tf.variable_scope('sub2'):
            var2 = tf.get_variable('var2', shape=[1])

    # 正确
    with tf.variable_scope('root') as scope:
        scope.reuse_variables()
        with tf.variable_scope('sub1'):
            var1 = tf.get_variable('var1', shape=[1])
        with tf.variable_scope('sub2'):
            var2 = tf.get_variable('var2', shape=[1])

    # 错误
    # with tf.variable_scope('root'):
    #     with tf.variable_scope('sub1') as scope:
    #         scope.reuse_variables()
    #         var1 = tf.get_variable('var1', shape=[1])
    #     with tf.variable_scope('sub2'):
    #         var2 = tf.get_variable('var2', shape=[1])


if __name__ == '__main__':
    # name_scope_and_variable_scope()
    # variable_scope_reuse()
    # variable_scope_init_reuse()
    # variable_scope_reuse_False()
    # name_scope_reuse()
    sub_variable_scope_reuse()