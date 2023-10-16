import collections
import itertools
import pickle
import time
from pathlib import Path

import nest_asyncio
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

import var

nest_asyncio.apply()

# import data_store
# from var import *
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[1:], 'GPU')
model = 'cifar10'
mnist = tf.keras.datasets.mnist
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
N = y_train.shape[0]
x_test_list = [x_test]
y_test_list = [y_test]


def nonIIDGen(beta=1):
    # class images into categories
    min_size = 0
    min_require_size = 10

    def record_net_data_stats(y_train, net_dataidx_map):
        net_cls_counts = {}
        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        # logger.info('Data statistics: %s' % str(net_cls_counts))
        return net_cls_counts

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(var.n_parties)]
        for k in range(var.K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, var.n_parties))
            # logger.info('proportions1: ', proportions)
            # logger.info('sum pro1:', np.sum(proportions))
            # Balance
            proportions = np.array([p * (len(idx_j) < N / var.n_parties)
                                    for p, idx_j in zip(proportions, idx_batch)])
            # logger.info('proportions2: ', proportions)
            proportions = proportions / proportions.sum()
            # logger.info('proportions3: ', proportions)
            proportions = (np.cumsum(proportions) *
                           len(idx_k)).astype(int)[:-1]
            # logger.info('proportions4: ', proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j,
            idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    net_dataidx_map = {}  # data map
    for j in range(var.n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    # change format of xtrain
    train_img = []
    train_label = []
    for j in range(var.n_parties):
        train_img.append([x_train[i] for i in net_dataidx_map[j]])
        train_label.append([y_train[i] for i in net_dataidx_map[j]])
    x_train_noniid = train_img
    y_train_noniid = train_label
    return x_train_noniid, y_train_noniid


def batch_format_fn(element):
    '''Flatten a batch `pixels` and return the features as an `OrderedDict`.'''
    if model == 'cifar10':
        return collections.OrderedDict(x=element['pixels'], y=element['label'])
    elif model == 'mnist':
        return collections.OrderedDict(x=tf.expand_dims(element['pixels'], -1), y=element['label'])


def preprocess(dataset, batchSize, batchCount, seednumber):
    print(1)
    return dataset.repeat(var.NUM_EPOCHS).shuffle(var.SHUFFLE_BUFFER, seed=seednumber).batch(
        batchSize).take(batchCount).map(batch_format_fn).prefetch(var.PREFETCH_BUFFER)


def make_federated_data(client_data, client_label, tnt, seednumber=1):
    temp_list = []
    if tnt == 'train':
        batchSize, batchCount = var.batchSizeTrain, var.batchCountTrain
    elif tnt == 'test':
        batchSize, batchCount = var.batchSizeTest, var.batchCountTest
    temp_list = [preprocess(
        tf.data.Dataset.from_tensor_slices(
            {'pixels': client_data[i], 'label': client_label[i]}),
        batchSize, batchCount, seednumber)
        for i in range(len(client_data))]
    return temp_list




print('preparing FL testing data')
federated_test_data = make_federated_data(x_test_list, y_test_list, 'test')



def create_keras_model():
    if model == 'cifar10':
        structure = tf.keras.models.Sequential([

            tf.keras.layers.InputLayer(input_shape=[32, 32, 3]),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(
                3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(
                3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(
                3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(
                3, 3), padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.75),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=10, activation='softmax')
        ])
    elif model == 'mnist':
        structure = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=[28, 28, 1]),  #
            # tf.keras.layers.InputLayer(input_shape=[32, 32, 3]),
            tf.keras.layers.Conv2D(filters=6, kernel_size=(
                5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(
                5, 5), padding='valid', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=120, activation='relu'),
            tf.keras.layers.Dense(units=84, activation='relu'),
            tf.keras.layers.Dense(units=10, activation='softmax')
        ])
    return structure


def model_fn():
    keras_model = create_keras_model()
    # keras_model.summary()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=federated_test_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


evaluation = tff.learning.build_federated_evaluation(
    model_fn, use_experimental_simulation_loop=True)
iterative = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))
iteratives = []
state = iterative.initialize()


def fedAvg(state, federated_train_data, NUM_SELECTED,
           iterative):  # fix the number of clients
    record_metrics = []
    record_valid = []
    record_time = []
    for round_num in range(1, var.NUM_ROUNDS + 1):
        selected_client_index = np.random.choice(
            var.clientCount, NUM_SELECTED, replace=False)
        record_time.append(max(var.latency[selected_client_index]))
        selected_client_data = [federated_train_data[i]
                                for i in selected_client_index]
        # with tf.profiler.experimental.Profile('multigpu'):
        state, metrics = iterative.next(state, selected_client_data)
        record_metrics.append(metrics['train']['sparse_categorical_accuracy'])
        print('FedAvg {:2d}, {:2d} clients, Batch size {:2d}, {:2d} Epoch, metrics={}'.
              format(round_num, len(selected_client_index), var.batchSizeTrain,
                     var.NUM_EPOCHS, list(metrics.items())[-2:]),
              flush=True, file=f)
        print('FedAvg {:2d}, {:2d} clients, Batch size {:2d}, {:2d} Epoch, metrics={}'.
              format(round_num, len(selected_client_index), var.batchSizeTrain, var.NUM_EPOCHS,
                     list(metrics.items())[-2:]))
        if round_num % var.TEST_frequency == 1:
            print(time.strftime("%Y%m%d %H%M%S"), flush=True, file=f)
            print(time.strftime("%Y%m%d %H%M%S"))
            metrics_valid = evaluation(state.model, federated_test_data)
            record_valid.append(
                metrics_valid['eval']['sparse_categorical_accuracy'])
            print('FedAvg ' + str(round_num) + ', validation',
                  metrics_valid, flush=True, file=f)
            print('FedAvg ' + str(round_num) + ', validation', metrics_valid)
            # record_valid[round_for_valid - 1] = metrics_valid['sparse_categorical_accuracy']
    # return record_valid,record_time
    return record_valid


# tiercount start from 1, corresponding tier start from 0
def fedAor(state, federated_train_data, one, clientTier, iteratives):
    tierCount = len(clientTier) if one['utiers'] == 0 else one['utiers']
    states = [state] * tierCount
    # selected_client_index = clientTier
    selected_client_data = [[federated_train_data[i]
                             for i in tierSet] for tierSet in clientTier]
    record_valid = []
    for round_num in range(1, var.NUM_ROUNDS + 1):
        flag = [0] * tierCount  # count the number of tiers involved
        print('tierCount' + str(tierCount), flush=True, file=f)
        print('var.clientTier' + str(len(var.clientTier)), flush=True, file=f)
        weights = [len(clientTier[i]) for i in range(tierCount)]
        for tier in range(tierCount):
            # determine whether the current need to be calculated (multipliler and clients exist)
            if (round_num % (tier + 1) == 0 and weights[tier] != 0):
                flag[tier] = 1
                # in there is some client in the tier, then do something
                # with tf.profiler.experimental.Profile('multigpu'):
                print('tier' + str(tier), flush=True, file=f)
                if tier < int(1 / var.lr):
                    states[tier], metrics = iteratives[tier].next(
                        states[tier], selected_client_data[tier])
                else:
                    states[tier], metrics = iteratives[-1].next(
                        states[tier], selected_client_data[tier])
                flagSum = sum(flag)
                if flagSum == 1:
                    globalState = states[tier]
                    if weights[tier] != 1:
                        for layer in range(len(globalState.model.trainable)):
                            globalState.model.trainable[layer] = (
                                    globalState.model.trainable[layer] * weights[tier])
                elif flagSum > 1:
                    for layer in range(len(globalState.model.trainable)):
                        globalState.model.trainable[layer] = (globalState.model.trainable[layer]
                                                              + states[tier].model.trainable[layer] * weights[tier])
        weightSum = np.sum(weights, where=flag)
        if weightSum >= 2:
            for layer in range(len(globalState.model.trainable)):
                globalState.model.trainable[layer] = globalState.model.trainable[layer] / weightSum
            for tier in range(tierCount):
                if flag[tier]:
                    states[tier] = globalState
                #     for layer in range(len(globalState.model.trainable)):
                #         states[tier].model.trainable[layer] = globalState.model.trainable[layer]
                # #states[tier].model.trainable=globalState.model.trainable
        # print('Rep'+str(re)+'Agg'+str(tierCount)+'round {:2d}, {:2d} clients, Batch size {:2d}, {:2d} Epoch, metrics={}'.format(round_num,len(selected_client_index),batchSizeTrain,NUM_EPOCHS, list(metrics.items())[-2:]))
        print('Agg' + str(tierCount) + 'round' + str(round_num) + 'metrics' + str(
            list(metrics.items())),
              flush=True, file=f)
        print('Agg' + str(tierCount) + 'round' +
              str(round_num) + 'metrics' + str(list(metrics.items())), flush=True)
        if round_num % var.TEST_frequency == 1:
            metrics_valid = evaluation(globalState.model, federated_test_data)
            record_valid.append(
                metrics_valid['eval']['sparse_categorical_accuracy'])
            print('Agg ' + str(tierCount) + ' validation',
                  metrics_valid, flush=True, file=f)
            print(time.strftime("%Y%m%d %H%M%S"), flush=True, file=f)
            print('Agg ' + str(tierCount) + ' validation',
                  metrics_valid, flush=True)
            print(time.strftime("%Y%m%d %H%M%S"), flush=True)
    return record_valid


# grid initialization
grid = []
methods_with_repetion = ['aor']
methods_without_repetion = ['avg']
utiers = [0, 1, 2, 3]

# # compare differnt methods in iid
grid.append({
    'betaValues': [10],
    'tierTimes': var.tierTimes[:2],
    'utiers': [0, 1],
    'methods': methods_with_repetion,
    'model': [model]
})

# impact of non-iid
grid.append({
    'betaValues': [0.1, 1, 10],
    'tierTimes': var.tierTimes[:2],
    'utiers': [0, 1],
    'methods': methods_with_repetion,
    'model': [model]
})

# imact of tier time
grid.append({
    'betaValues': [0.1, 1],
    'tierTimes': [10, 20, 60],
    'utiers': [0],
    'methods': methods_with_repetion,
    'model': [model]
})

# # fedavg on non-IID
grid.append({
    'betaValues': [0.1, 1],
    'methods': methods_without_repetion,
    'model': [model]
})

# impact of utlized tiers
grid.append({
    'repetion': range(var.repetion),
    'betaValues': [0.1, 10],
    'tierTimes': [var.tierTimes[0]],
    'utiers': utiers,
    'methods': methods_with_repetion,
    'model': [model]
})

# imact of tier time

setting_list = []
for i, onegrid in enumerate(grid):
    setting_list += [dict(zip(onegrid.keys(), values))
                     for values in itertools.product(*onegrid.values())]
setting_list1 = setting_list
for i, d1 in reversed(list(enumerate(setting_list))):
    for j, d2 in enumerate(setting_list[0:i - 1]):
        if i != 0 and d1 == d2:
            setting_list.pop(i)


def train(one):
    print(one, flush=True, file=f)
    print(one)
    iteratives = []
    for i in range(int(1 / var.lr)):
        print(var.lr * (i + 1))
        iteratives.append(tff.learning.build_federated_averaging_process(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(
                learning_rate=min(1, var.lr * (i + 1))),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(
                learning_rate=1),
            use_experimental_simulation_loop=True))
        if one['methods'] == 'avg' or (one['methods'] == 'aor' and one['utiers'] == 1):
            break

    iterative = iteratives[0]
    state = iterative.initialize()

    x_train_noniid, y_train_noniid = nonIIDGen(beta=one['betaValues'])
    federated_train_data = make_federated_data(
        x_train_noniid, y_train_noniid, 'train', seednumber=1)

    if one['methods'] == 'avg':
        return fedAvg(state, federated_train_data, var.clientCount, iteratives[0])
    elif one['methods'] == 'aor':
        clientTier, latency = var.tierGenerator(one['tierTimes'])
        return fedAor(state, federated_train_data, one, clientTier, iteratives)


pathname = 'result/fig5_8'
p = Path(pathname + '/')
p.mkdir(exist_ok=True)

file_name = './' + pathname + '/' + '3kforall'
f = open(file_name + '.txt', 'w')

data = []
actlist = []
seq = 0
for rep in range(var.repetion):
    latency = var.latencyGenerator()
    iteratives = []
    for i in range(int(1 / var.lr)):
        print(var.lr * (i + 1))
        iteratives.append(tff.learning.build_federated_averaging_process(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(
                learning_rate=min(1, var.lr * (i + 1))),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(
                learning_rate=1),
            use_experimental_simulation_loop=True))
    iterative = iteratives[0]
    state = iterative.initialize()

    fdatas = []
    for _, beta in enumerate(var.betaValues):
        print('beta' + str(beta))
        x_train_noniid, y_train_noniid = nonIIDGen(beta)
        fdatas.append(make_federated_data(
            x_train_noniid, y_train_noniid, 'train', seednumber=1))
    for ione, one in enumerate(setting_list):
        betaindex = var.betaValues.index(one['betaValues'])
        actlist.append(one)
        print(one, flush=True, file=f)
        print(one)
        if one['methods'] == 'avg' and rep == 0:
            data.append(fedAvg(state, fdatas[betaindex], var.clientCount, iterative))
        elif one['methods'] == 'aor':
            clientTier = var.tierGenerator_latency(latency, one['tierTimes'])
            onedata = fedAor(state, fdatas[betaindex], one, clientTier, iteratives)
            data.append(onedata)

        with open(file_name + '_' + str(seq) + '.pkl', 'wb') as pk:
            pickle.dump(onedata, pk)
            pickle.dump(one, pk)
            pk.close()
        seq = seq + 1

with open(file_name + '.pkl', 'wb') as pk:
    pickle.dump(data, pk)
    pickle.dump(actlist, pk)
    pk.close()
