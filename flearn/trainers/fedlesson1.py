import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import math
from .myfedbase import myBaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad
from var import lessonTaskAssign
from datetime import datetime
import copy
import math

class Server(myBaseFedarated):

    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train')
        self.clientTier, self.actTrain = lessonTaskAssign(params, proxtddFlag=0)
        #self.actTrain = [params['num_epochs'] if i > params['num_epochs'] else i for i in self.actTrain]

        #self.actTrain = [math.ceil(i/params['batch_size']) for i in self.actTrain]
        #self.actTrain = [10 for i in self.actTrain]
        self.tierCount = len(self.clientTier)
        self.tierClientCount = [len(tier) for tier in self.clientTier]
        try:
            self.stale_decay = params['stale_decay']
            self.epoch_decay = params['epoch_decay']
        except:
            self.stale_decay = self.epoch_decay = 1
        # self.inner_opt = PerturbedGradientDescent(params['learning_ralowTierPriorityte'], params['mu'])
        # self.stale_learning_rate = [params['learning_rate'] * (i + 1) * self.stale_decay ** (i) for i in
        #                             range(self.tierCount)]
        try:
            self.logbase = params['logbase']
        except:
            self.logbase=1.45
        self.stale_learning_rate = [
            min(max(1,math.log((i+1),self.logbase))*params['learning_rate'],0.1)
            for i in range(self.tierCount)]

        #self.stale_learning_rate = [params['learning_rate']*params['expbase']**(-i) for i in range(self.tierCount)]
        self.inner_opts = [PerturbedGradientDescent(lr, params['mu']) for lr in self.stale_learning_rate]
        super(Server, self).__init__(params, learner, dataset, self.clientTier)
        # model containers for different tiers
        # self.latest_models = [self.latest_model]*self.tierCount

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        # testing acc, training acc, training loss
        record = [[] for _ in range(3)]
        for i in range(self.num_rounds):
            # test while train
            if i % self.eval_every == 0:

                stats_train = self.train_error_and_loss()
                stats = self.test()  # have set the latest model for all clients
                print(
                    'At round {} accuracy: {}\n'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  # testing accuracy
                record[0].append(np.sum(stats[3]) * 1.0 / np.sum(stats[2]))

                print('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(
                    stats_train[2])))
                print('At round {} training loss: {}'.format(i,
                                                             np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(
                                                                 stats_train[2])))

                record[1].append(np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2]))
                record[2].append(np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2]))

            # buffer for receiving client solutions
            csolns = []
            tierInvolved = [False] * self.tierCount

            # for tier, cs in enumerate(self.clientTier):
            for tier, cs in enumerate(self.clientTier[0:3]):
                if (i + 1) % (tier + 1) == 0 and bool(cs):
                    tierInvolved[tier] = True
                    self.inner_opts[tier].set_params(self.latest_models[tier], self.tier_models[tier])
                    # weightTierCoe = (tier + 1) * self.stale_decay ** tier
                    weightTierCoe = 1
                    for idx, c in enumerate([self.clients[k] for k in cs]):
                        # communicate the latest model client set_params
                        # client.set_params  -->  model.set_params
                        c.set_params(self.latest_models[tier])
                        # solve minimization locally

                        #soln, stats = c.solve_sgd(num_epochs=self.actTrain[cs[idx]], batch_size=self.batch_size)
                        soln = c.solve_bgd(num_allowance=self.actTrain[cs[idx]], batch_size=self.batch_size)



                        # gather solutions from client
                        csolns.append(soln)

                        # track communication cost
                        #self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            if csolns:
                # self.latest = self.aggregate(csolns)
                value = self.aggregate(csolns)

                nsn = []
                for v1, v2 in zip(self.latest_model, value):
                    nsn.append(v2 + v1)
                self.latest_model = nsn

                # self.latest_model_new = self.aggregate(csolns)
                # importance=sum([a*b for a, b in zip(self.tierClientCount , tierInvolved)])
                # self.latest_model=[(a-b) *importance/30 for a,b in zip(self.latest_model_new,self.latest_model) ]

            #  model.set_params
            for tier, cs in enumerate(self.clientTier):
                if tierInvolved[tier] == 1:
                    # self.latest_models[tier] = copy.deepcopy(self.latest_model)
                    self.latest_models[tier] = copy.deepcopy(self.latest_model)
                    # self.tier_models[tier].set_params(self.latest_models[tier])

        # # final test model
        # stats = self.test()
        # stats_train = self.train_error_and_loss()
        # self.metrics.accuracies.append(stats)
        # self.metrics.train_accuracies.append(stats_train)
        # print('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        # print('At round {} training accuracy: {}'.format(self.num_rounds,
        #                                                       np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
        # record.append([np.sum(stats[3]) * 1.0 / np.sum(stats[2]),
        #                np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2]),
        #                0
        #                ])

        # date = datetime.now().strftime("%m_%d-%I:%M")
        # np.savez("lesson"+str(self.tierTime)+"data.pkl"+f"{date}.npz",
        #          data.pkl=record,
        #          )
        return record
