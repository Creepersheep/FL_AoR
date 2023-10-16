import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from flearn.trainers.fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from datetime import datetime


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        record = [[] for _ in range(3)]
        gpus=tf.config.experimental.list_physical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy(gpus)
        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                tqdm.write(
                    'At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(
                    stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i,
                                                                  np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(
                                                                      stats_train[2])))
                # record.append([np.sum(stats[3]) * 1.0 / np.sum(stats[2]),
                #                np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2]),
                #                np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])
                #                ])
                record[0].append(np.sum(stats[3]) * 1.0 / np.sum(stats[2]))
                record[1].append(np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2]))
                record[2].append(np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2]))

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)),
                                              replace=False)

            csolns = []  # buffer for receiving client solutions

            for idx, c in enumerate(active_clients.tolist()):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                with strategy.scope():
                    soln, stats = c.solve_sgd(num_epochs=self.num_epochs, batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(soln)

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            self.latest_model = self.aggregate(csolns)

        # # final test model
        # stats = self.test()
        # stats_train = self.train_error_and_loss()
        # self.metrics.accuracies.append(stats)
        # self.metrics.train_accuracies.append(stats_train)
        # tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        # tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
        # record.append([np.sum(stats[3]) * 1.0 / np.sum(stats[2]),
        #                np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2]),
        #                0
        #                ])
        # date = datetime.now().strftime("%m_%d-%I:%M")
        # np.savez("avg" + str(self.tierTime) + "data.pkl" + f"{date}.npz",
        #          data.pkl=record,
        #          )
        return record