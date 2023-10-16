import numpy as np
import tensorflow as tf
from tqdm import trange

from flearn.utils.model_utils import batch_data, batch_data_multiple_iters, epoch_batch_data
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class Model(object):
    '''
    Assumes that images are 28px by 28px
    '''

    # def __init__(self, num_classes, optimizer, seed=1):
    #
    #     # params
    #     self.num_classes = num_classes
    def __init__(self, params, optimizer, seed=1):

        # params
        self.num_classes = params['model_params'][0]
        # try:
        #     self.regValue=params['regValue']
        # except:
        #     self.regValue=None

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(
                optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops


    def create_model(self, optimizer):
        """Model function for Logistic Regression."""


        features = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='features')
        labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')



        l01 = tf.layers.conv2d(inputs=features, filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        l02 = tf.layers.conv2d(inputs=l01, filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        l03 = tf.layers.max_pooling2d(inputs=l02, pool_size=(2, 2), strides=(2, 2))
        l03d = tf.layers.dropout(inputs=l03,rate=0.2)
        l05 = tf.layers.conv2d(inputs=l03d, filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        l06 = tf.layers.conv2d(inputs=l05, filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        l07 = tf.layers.max_pooling2d(inputs=l06, pool_size=(2, 2), strides=(2, 2))
        l07d = tf.layers.dropout(inputs=l07, rate=0.2)
        l09 = tf.layers.conv2d(inputs=l07d, filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        l10 = tf.layers.conv2d(inputs=l09, filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        l11 = tf.layers.max_pooling2d(inputs=l10, pool_size=(2, 2), strides=(2, 2))
        l12 = tf.layers.flatten(inputs=l11)
        l12d = tf.layers.dropout(inputs=l12, rate=0.2)
        l15 = tf.layers.dense(inputs=l12d, units=128, activation='relu')
        l15d = tf.layers.dropout(inputs=l15, rate=0.2)


        logits = tf.layers.dense(inputs=l15d, units=self.num_classes, activation='relu')
        #logits=tf.layers.dense(inputs=l16, units=self.num_classes, activation='softmax')

        predictions = {
            # return the index of the largest value for argmax
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits= logits)
        # predictions = {
        #     # return the index of the largest value for argmax
        #     "classes": tf.argmax(input=logits, axis=1),
        #     "probabilities": logits
        # }
        #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        # compare two values (categories)
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):

        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                                        feed_dict={self.features: data['x'], self.labels: data['y']})
            grads = process_grad(model_grads)

        return num_samples, grads

    # sover_inner goes through the entire dataset
    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem'''
        a = []
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):

            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                                  feed_dict={self.features: X, self.labels: y})
                a.append(self.get_params())

        soln = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return soln, comp

    def solve_sgd(self, data, num_epochs, batch_size):
        '''Solves local optimization problem'''
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                                  feed_dict={self.features: X, self.labels: y})
                break
        soln = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return soln, comp

    def solve_bgd(self, data, allowance=100, batch_size=32):
        '''Solves local optimization problem'''

        for X, y in epoch_batch_data(data, allowance, batch_size):
            with self.graph.as_default():
                self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
            # with self.graph.as_default():
            #     _, tot_correct, loss = self.sess.run([self.train_op, self.eval_metric_ops, self.loss],
            #                                       feed_dict={self.features: data['x'], self.labels: data['y']})
        soln = self.get_params()
        comp = allowance * self.flops
        return soln, comp

    def solve_iters(self, data, num_iters=1, batch_size=32):
        '''Solves local optimization problem'''

        for X, y in batch_data_multiple_iters(data, batch_size, num_iters):
            with self.graph.as_default():
                self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = 0
        return soln, comp

    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss

    def close(self):
        self.sess.close()
