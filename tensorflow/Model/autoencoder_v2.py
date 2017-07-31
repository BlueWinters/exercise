# tensorflow version 1.10

import tensorflow as tf
import os
import time

from tensorflow.examples.tutorials.mnist import input_data

class AutoencoderV2(object):
    def __init__(self, encoder=[28*28], z_dim=400, decoder=[28*28],
                 batch_size=100, num_epochs=100):
        self.encoder = encoder
        self.z_dim = z_dim
        self.decoder = decoder
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.sess = tf.Session()
        self.learn_rate = 1e-3

        self._init_layer_set()
        self._init_loss()

    def __del__(self):
        self.sess.close()

    def _init_loss(self):
        shape = [self.batch_size, self.encoder[0]]
        self.x = tf.placeholder(tf.float32, shape, name='input')

        h = self.x
        for n in range(len(self.layer_set)-1):
            in_dim, out_dim = self.layer_set[n], self.layer_set[n+1]
            h = self._feedward(h, in_dim, out_dim, 'layer{}'.format(n))
        self._summary_mnist(h)

        with tf.name_scope('loss') as scope:
            square_loss = tf.reduce_sum(tf.square(self.x - h), axis=[1])
            self.loss = tf.reduce_mean(square_loss)
            tf.summary.scalar('loss', self.loss)

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.name_scope('train') as scope:
            self.trainer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss, var_list=self.vars)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./summary', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _init_layer_set(self):
        self.layer_set = self.encoder[:] # copy, not use '=' for assign
        self.layer_set.append(self.z_dim)
        self.layer_set.extend(self.decoder)

    def _feedward(self, input, in_dim, out_dim, name):
        shape = tf.shape(input)
        assert shape[-1] != in_dim

        with tf.variable_scope(name) as scope:
            W = tf.get_variable(name='W', shape=[in_dim, out_dim], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable(name='b', shape=[out_dim], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer())
            a = tf.nn.sigmoid(tf.matmul(input, W) + b)
            self._summary(W, b, a)
        return a

    def _summary(self, W, b, a):
        with tf.name_scope('filter') as scope:
            image = tf.reshape(W, [-1, 28, 28, 1])
            tf.summary.image('W', image, 100)
        with tf.name_scope('active') as scope:
            tf.summary.histogram('a', a)
        with tf.name_scope('bias') as scope:
            tf.summary.histogram('b', b)

    def _summary_mnist(self, output):
        with tf.name_scope('output') as scope:
            mnist = tf.reshape(output, [-1, 28, 28, 1])
            tf.summary.image('mnist', mnist, 10)

    def train_on_mnist(self):
        mnist = input_data.read_data_sets("./mnist/", one_hot=True)
        total_batch = int(mnist.train.num_examples / self.batch_size)
        shape = [self.batch_size, self.encoder[0]]

        for epoch in range(self.num_epochs):
            average_loss = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(self.batch_size)
                batch_x = batch_x.reshape(shape)
                summary, loss, _ = self.sess.run([self.merged, self.loss, self.trainer],
                                                 {self.x:batch_x})
                average_loss += loss / total_batch
            self.writer.add_summary(summary, epoch)
            print("Epoch : {:d}/{:d}, Loss : {:9f}".format(epoch+1, self.num_epochs, average_loss))

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
    config = {'encoder':[28*28], 'decoder':400, 'z_dim':400,
              'batch_size':100, 'num_epochs':100}

    autoencoder = AutoencoderV2()
    autoencoder.train_on_mnist()
