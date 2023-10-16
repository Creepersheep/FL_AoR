import numpy as np
import tensorflow as tf
from tqdm import tqdm

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad


# 3 models
# tier_models[client_model]: models for different tier with different
    # inner_opts: different lr for tier_models
# client_model: model for the global model for test, bound with optimizer
    # inner_opt: optimizer for client_model
# clients: list of models for each client
# lastest_models[  ]: lastest model corresponding to tiers
# lastest_model: global model matrix, value passing


class myBaseFedarated(object):
    def __init__(self, params, learner, dataset, clientTier):

        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);
        # create worker nodes
        tf.reset_default_graph()
        # set different learning rate for tiers

        # version same learning rate 0914
        # version adjusted learning rate 0915
        self.tier_models = [learner(params, self.inner_opts[tier], self.seed+tier) for tier in range(self.tierCount)]
        self.clients = self.setup_clients(dataset, clientTier, self.tier_models)
        print('{} Clients in Total'.format(len(self.clients)))
        # self.latest_model = self.client_model.get_params()
        self.latest_models = [self.tier_models[0].get_params() for tier in range(self.tierCount)]
        # initialize system metrics
        self.metrics = Metrics(self.clients, params)

        # original lastmodel and clientmodel are kept for gradshow and test
        # self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)

        self.client_model = learner(params, self.inner_opts[0], self.seed)
        self.latest_model = self.client_model.get_params()


    def __del__(self):
        # self.client_model.close()
        for i in range(len(self.tier_models)):
            self.tier_models[i].close()

    def setup_clients(self, dataset, clientTier, models=None):
        '''instantiates clients based on given train and test data.pkl directories

        Return:
            list of Clients
        '''
        # models=[]
        # for i in range(self.tierCount):
        #     self.inner_opt._lr=params['learning_rate']*(i+1)
        #     models.append(learner(*params['model_params'], self.inner_opt, self.seed))
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        tiers = [None for _ in users]
        for t, singleTierClients in enumerate(clientTier):
            for j in singleTierClients:
                tiers[j] = t
        # model=learner(*params['model_params'], self.inner_opt, self.seed)
        all_clients = [Client(u, g, train_data[u], test_data[u], models[t]) for u, g, t in zip(users, groups, tiers)]
        return all_clients

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        each_correct=[]

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            each_correct.append(ct)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses, each_correct

    def show_grads(self):
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)

        intermediate_grads = []
        samples = []

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model)
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples))
        intermediate_grads.append(global_grads)

        return intermediate_grads

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass

    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients

        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        '''

        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return indices, np.asarray(self.clients)[indices]

    def aggregate(self, wsolns):
        total_weight = 0.0
        base = [0] * len(wsolns[0][1])
        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w * v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]
        #averaged_soln = [v / 50000 for v in base]
        return averaged_soln
