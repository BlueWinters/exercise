# coding=utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim

def depthwise_conv2d():
    # 参照：https://blog.csdn.net/mao_xiao_feng/article/details/78002811
    img1 = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
    img2 = tf.constant(value=[[[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]]]],dtype=tf.float32)
    img = tf.concat(values=[img1,img2],axis=3)
    filter1 = tf.constant(value=0, shape=[3,3,1,1],dtype=tf.float32)
    filter2 = tf.constant(value=1, shape=[3,3,1,1],dtype=tf.float32)
    filter3 = tf.constant(value=2, shape=[3,3,1,1],dtype=tf.float32)
    filter4 = tf.constant(value=3, shape=[3,3,1,1],dtype=tf.float32)
    filter_out1 = tf.concat(values=[filter1,filter2],axis=2)
    filter_out2 = tf.concat(values=[filter3,filter4],axis=2)
    filter = tf.concat(values=[filter_out1,filter_out2],axis=3)

    point_filter = tf.constant(value=1, shape=[1,1,4,4],dtype=tf.float32)

    # 1.depthwise_conv2d
    out_img = tf.nn.depthwise_conv2d(input=img, filter=filter, strides=[1,1,1,1],rate=[1,1], padding='VALID')
    # 2.pointwise_conv2d
    out_img = tf.nn.conv2d(input=out_img, filter=point_filter, strides=[1,1,1,1], padding='VALID')

    # depthwise部分的另一种用法
    # out_img = tf.nn.separable_conv2d(input=img, depthwise_filter=filter, pointwise_filter=point_filter,
    #                                  strides=[1, 1, 1, 1], rate=[1, 1], padding='VALID')


    with tf.Session() as sess:
        print(sess.run(out_img))

    # 输出：
    # [[[[72.  72.  72.  72.]
    #    [90.  90.  90.  90.]]
    #
    #  [[72.  72.  72.  72.]
    #     [90.  90.  90.  90.]]]]

def depthwise_conv2d_helper():
    img1 = tf.constant(value=[[[[1], [2], [3], [4]], [[1], [2], [3], [4]], [[1], [2], [3], [4]], [[1], [2], [3], [4]]]], dtype=tf.float32)
    img2 = tf.constant(value=[[[[1], [2], [3], [4]], [[1], [2], [3], [4]], [[1], [2], [3], [4]], [[1], [2], [3], [4]]]], dtype=tf.float32)
    img3 = tf.constant(value=[[[[1], [2], [3], [4]], [[1], [2], [3], [4]], [[1], [2], [3], [4]], [[1], [2], [3], [4]]]], dtype=tf.float32)
    img = tf.concat(values=[img1, img2, img3], axis=3)
    filter1 = tf.constant(value=1, shape=[3, 3, 1, 1], dtype=tf.float32)
    filter2 = tf.constant(value=2, shape=[3, 3, 1, 1], dtype=tf.float32)
    filter3 = tf.constant(value=2, shape=[3, 3, 1, 1], dtype=tf.float32)
    filter = tf.concat(values=[filter1, filter2, filter3], axis=2)

    out_img = tf.nn.depthwise_conv2d(input=img, filter=filter, strides=[1, 1, 1, 1], rate=[1, 1], padding='VALID')
    print(out_img.get_shape().as_list())

    with tf.Session() as sess:
        print(sess.run(out_img))



if __name__ == '__main__':
    # depthwise_conv2d()
    depthwise_conv2d_helper()