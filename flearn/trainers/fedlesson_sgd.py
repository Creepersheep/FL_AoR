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


class Server(myBaseFedarated):

    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train')
        self.clientTier, self.actTrain = lessonTaskAssign(params, proxtddFlag=0)
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
        # self.stale_learning_rate = [
        #     min(max(1, math.log(i + 1, params['logbase'])) * params['learning_rate'], 0.5)
        #     for i in range(self.tierCount)]
        #self.stale_learning_rate = [params['learning_rate'] for i in range(self.tierCount)]
        if params['lr_mode'] == 0:
            self.stale_learning_rate = [
                min(max(1, math.log(i + 1, params['logbase'])) * params['learning_rate'], 0.5)
                for i in range(self.tierCount)]
        elif params['lr_mode'] == 1:
            self.stale_learning_rate = [params['learning_rate'] for _ in range(self.tierCount)]
        elif params['lr_mode'] == 2:
            self.stale_learning_rate = [params['learning_rate'] * (i + 1) for i in range(self.tierCount)]
        self.inner_opts = [PerturbedGradientDescent(lr, params['mu']) for lr in self.stale_learning_rate]
        super(Server, self).__init__(params, learner, dataset, self.clientTier)
        # model containers for different tiers
        # self.latest_models = [self.latest_model]*self.tierCount

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        # testing acc, training acc, training loss
        record = [[] for _ in range(5)]
        for i in range(self.num_rounds):
            # test while train
            if i % self.eval_every == 0:
                stats = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()
                print('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  # testing accuracy
                print('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(
                    stats_train[2])))
                print('At round {} training loss: {}'.format(i,
                                                             np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(
                                                                 stats_train[2])))
                each_rightness = [round(i / j, 3) for i, j in zip(stats_train[5], stats_train[2])]
                print(each_rightness)
                record[0].append(np.sum(stats[3]) * 1.0 / np.sum(stats[2]))
                record[1].append(np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2]))
                record[2].append(np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2]))
                record[3].append(each_rightness)
            # buffer for receiving client solutions
            csolns = []
            tierInvolved=[]
            clientInvolved=[]
            bestacc = max(record[3][-1])
            weights = [bestacc / i for i in record[3][-1]]

            for tier, cs in enumerate(self.clientTier):
                if (i + 1) % (tier + 1) == 0 and bool(cs):
                    tierInvolved.append(tier)
                    self.inner_opts[tier].set_params(self.latest_models[tier], self.tier_models[tier])
                    # weightTierCoe = (tier + 1) * self.stale_decay ** tier

                    for idx, c in enumerate([self.clients[k] for k in cs]):
                        # communicate the latest model client set_params
                        # client.set_params  -->  model.set_params
                        c.set_params(self.latest_models[tier])
                        # solve minimization locally
                        # soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                        soln, stats = c.solve_sgd(num_epochs=self.actTrain[cs[idx]], batch_size=1)




                        #soln[0] = soln[0] * weights[cs[idx]]




                        # gather solutions from client
                        csolns.append(soln)

                        # track communication cost
                        self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            if csolns:
                # self.latest = self.aggregate(csolns)
                self.latest_model = self.aggregate(csolns)

                # self.latest_model_new = self.aggregate(csolns)
                # importance=sum([a*b for a, b in zip(self.tierClientCount , tierInvolved)])
                # self.latest_model=[(a-b) *importance/30 for a,b in zip(self.latest_model_new,self.latest_model) ]

            #  model.set_params
            for tier in tierInvolved:
                self.latest_models[tier] = copy.deepcopy(self.latest_model)
                clientInvolved+=self.clientTier[tier]
            print(tierInvolved)
            print(np.sort(clientInvolved))
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
