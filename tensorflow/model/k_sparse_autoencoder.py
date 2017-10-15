# tensorflow version 1.10

import tensorflow as tf
import os
import time

from tensorflow.examples.tutorials.mnist import input_data

class KSparseAutoencoder(object):
    def __init__(self, sess, encoder, z_dim, decoder, k=0.1, name='Autoencoder'):
        self.encoder = encoder
        self.z_dim = z_dim
        self.decoder = decoder
        self.k = k
        self.name = name
        # session
        self.sess = sess

        self._init_vars()
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def _init_vars(self):
        with tf.variable_scope(self.name) as vs:
            self._init_encoder_vars()
            self._init_decoder_vars()

    def _init_encoder_vars(self):
        in_list = self.encoder[:]
        out_list = self.encoder[1:]
        out_list.append(self.z_dim)
        with tf.variable_scope('encoder') as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                self._set_layer_vars(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))

    def _init_decoder_vars(self):
        in_list = [self.z_dim]
        in_list.extend(self.decoder[:-1])
        out_list = self.decoder[:]
        with tf.variable_scope('decoder') as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                self._set_layer_vars(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))

    def _set_layer_vars(self, in_dim, out_dim, name, stddev=0.1):
        with tf.variable_scope(name) as vs:
            k = tf.get_variable('W', [in_dim, out_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [out_dim],
                                initializer=tf.constant_initializer(0))
        return k, b

    def _get_W_square(self, name):
        with tf.variable_scope(name, reuse=True) as vs:
            W = tf.get_variable('W')
        return tf.reduce_sum(tf.square(W))

    def _feedward(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            a = tf.matmul(input, W) + b
        return tf.nn.sigmoid(a)

    def _get_ksparse(self, h):
        shape = tf.shape(h[-1])

        n_samples_int = shape[0]
        n_samples_float = tf.cast(n_samples_int, tf.float32) # cast: int32 --> float32
        k_float = tf.cast(self.k*n_samples_float, tf.int32)

        topk_h, _ = tf.nn.top_k(h[-1], k=k_float)
        topk_h_r = tf.reshape(topk_h, tf.stack([n_samples_int, 1]))
        drop = tf.where(h[-1] < topk_h_r,
                        tf.zeros(h.shape, tf.float32), tf.ones(h.shape, tf.float32))
        # lifetime sparsity: keep most top k value
        return h[-1] * drop

    def loss(self, input):
        h = []
        h.append(input)

        with tf.variable_scope(self.name, reuse=True) as vs:
            # encoder
            n_encoder = len(self.encoder)
            with tf.variable_scope('encoder', reuse=True) as vs:
                for n in range(n_encoder):
                    h.append(self._feedward(h[-1], 'layer_'+str(n)))

            # k-sparse penalty
            h[-1] = self._get_ksparse(h[-1])

            # decoder
            n_decoder = len(self.decoder)
            with tf.variable_scope('decoder', reuse=True) as vs:
                for n in range(n_decoder):
                    h.append(self._feedward(h[-1], 'layer_'+str(n)))
        # return loss
        return tf.reduce_mean(tf.square(h[-1] - input) + sparse_penalty)

    def save(self, path):
        saver = tf.train.Saver(self.vars)
        saver.save(self.sess, path)

    def restore(self, path):
        saver = tf.train.Saver(self.vars)
        saver.restore(self.sess, path)

    def visual_filter(self):
        pass


if __name__ == '__main__':
    # set parameters
    encoder = [28*28]
    z_dim = 1000
    decoder = [28*28]
    num_epochs = 100
    batch_size = 100
    learn_rate = 1e-3
    shape = [batch_size, 28*28]

    # construct model
    sess = tf.Session()
    ae = KSparseAutoencoder(sess, encoder=encoder, z_dim=z_dim, decoder=decoder)
    input = tf.placeholder(tf.float32, shape)
    # initialize model
    loss = ae.loss(input)

    # set optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    train = optimizer.minimize(loss, var_list=ae.vars)
    sess.run(tf.global_variables_initializer())

    # read data
    mnist = input_data.read_data_sets("mnist/", one_hot=True)

    # train model
    start_time = time.time()
    for epoch in range(num_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)
        avg_loss = 0
        for i in range(total_batch):
            batch_x, _ = mnist.train.next_batch(batch_size)

            batch_x = batch_x.reshape(shape)
            l, _ = sess.run([loss, train], {input: batch_x})
            avg_loss += l / total_batch

        print("Epoch : {:04d}, Loss : {:9f}".format(epoch + 1, avg_loss))
    print("Training time : {}".format(time.time() - start_time))

    # save model
    ckpt_dir = 'ckpt/'
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    ae.save('ckpt/model.ckpt')

    # close session
    sess.close()