import tensorflow as tf
import numpy as np

import nest_asyncio
from tempfile import TemporaryFile
nest_asyncio.apply()

from collections import *
import tensorflow_federated as tff
import random
from matplotlib import pyplot as plt
import math
import sys
import time
#import data_store
import shelve
import collections
from var import *


model='cifar10'
#model='mnist'
mnist = tf.keras.datasets.mnist
cifar10=tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
N = y_train.shape[0]
x_test_list = [x_test]
y_test_list = [y_test]

def nonIIDGen(beta=1):
    min_size = 0
    min_require_size = 10
    def record_net_data_stats(y_train, net_dataidx_map):
        net_cls_counts = {}
        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        #logger.info('Data statistics: %s' % str(net_cls_counts))
        return net_cls_counts

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            # logger.info('proportions1: ', proportions)
            # logger.info('sum pro1:', np.sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            # logger.info('proportions2: ', proportions)
            proportions = proportions / proportions.sum()
            # logger.info('proportions3: ', proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info('proportions4: ', proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    net_dataidx_map = {} # data map
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    # change format of xtrain
    train=[]
    train_label=[]
    for j in range(n_parties):
        train.append([x_train[i] for i in net_dataidx_map[j]])
        train_label.append([y_train[i] for i in net_dataidx_map[j]])
    x_train_noniid=train
    y_train_noniid=train_label
    return x_train_noniid,y_train_noniid




def batch_format_fn(element):
    '''Flatten a batch `pixels` and return the features as an `OrderedDict`.'''
    if model=='cifar10':
        return collections.OrderedDict(x=element['pixels'], y=element['label'])
    elif model=='mnist':
        return collections.OrderedDict(x=tf.expand_dims(element['pixels'], -1), y=element['label'])

def preprocess(dataset,batchSize,batchCount,seednumber):
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=seednumber).batch(
      batchSize).take(batchCount).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

def make_federated_data(client_data, client_label,tnt,seednumber=1):
    temp_list=[]
    if tnt=='train':
      batchSize,batchCount=batchSizeTrain,batchCountTrain
    elif tnt=='test':
      batchSize, batchCount = batchSizeTest, batchCountTest
    # for x in range(n_parties):
    #     dataset1=tf.data.Dataset.from_tensor_slices({'pixels':client_data[x], 'label':client_label[x]})
    #     # dataset2 = preprocess(dataset1)
    #     # temp_list.append(dataset2)
    #     temp_list.append(preprocess(dataset1,batchSize,batchCount))
    temp_list = [preprocess(
      tf.data.Dataset.from_tensor_slices({'pixels': client_data[i], 'label': client_label[i]}),
      batchSize,batchCount,seednumber)
      for i in range(len(client_data))]
    return temp_list

#print('preparing FL training data')
#federated_train_data = make_federated_data(x_train, y_train,'train',seednumber=1)



print('preparing FL testing data')
federated_test_data = make_federated_data(x_test_list,y_test_list,'test')








cpu_device = tf.config.list_logical_devices('CPU')[0]
gpu_devices = tf.config.list_logical_devices('GPU')
tff.backends.native.set_local_python_execution_context(server_tf_device=cpu_device,
    client_tf_devices=gpu_devices,
    #clients_per_thread=1,
    max_fanout=100)



def create_keras_model():
    if model=='cifar10':
        structure=tf.keras.models.Sequential([
          #tf.keras.layers.InputLayer(input_shape=[28,28,1]),   #
          tf.keras.layers.InputLayer(input_shape=[32, 32, 3]),
          tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(units=120, activation='relu'),
          tf.keras.layers.Dense(units=84, activation='relu'),
          tf.keras.layers.Dense(units=10, activation='softmax')
      ])
    elif model=='mnist':
        structure = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=[28,28,1]),   #
            #tf.keras.layers.InputLayer(input_shape=[32, 32, 3]),
            tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=120, activation='relu'),
            tf.keras.layers.Dense(units=84, activation='relu'),
            tf.keras.layers.Dense(units=10, activation='softmax')
        ])
    structure.summary()
    return structure

def model_fn():
    keras_model = create_keras_model()
    #keras_model.summary()
    return tff.learning.from_keras_model(
      keras_model,
      input_spec=federated_test_data[0].element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

iteratives=[]
for i in range(clientCount):
    iteratives.append(tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1*(i+1)),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1),
    use_experimental_simulation_loop=True))
iterative=iteratives[0]
state = iterative.initialize()
evaluation = tff.learning.build_federated_evaluation(model_fn,use_experimental_simulation_loop=True)


def fedNum(state,NUM_SELECTED):     # fix the number of clients
    record_metrics=[]
    record_valid=[]
    record_time=[]
    for round_num in range(1, NUM_ROUNDS+1):
        selected_client_index = np.random.choice(clientCount, NUM_SELECTED, replace=False)
        record_time.append(max(latency[selected_client_index]))
        selected_client_data = [federated_train_data[i] for i in selected_client_index]
        state, metrics = iterative.next(state, selected_client_data)
        record_metrics.append(metrics['train']['sparse_categorical_accuracy'])
        print('5Cround {:2d}, {:2d} clients, Batch size {:2d}, {:2d} Epoch, metrics={}'.format(round_num,len(selected_client_index),batchSizeTrain,NUM_EPOCHS, list(metrics.items())[-2:]))
        if round_num % TEST_frequency == 1:
            print(time.strftime("%Y%m%d %H%M%S"))
            metrics_valid = evaluation(state.model, federated_test_data)
            record_valid.append(metrics_valid['sparse_categorical_accuracy'])
            print('validation', metrics_valid)
            #record_valid[round_for_valid - 1] = metrics_valid['sparse_categorical_accuracy']

    return record_valid,record_time

def fedAvgTier(state,rept,tierCount=len(clientTier)):     #  tiercount start from 1, corresponding tier start from 0
    record_valid=[]
    states=[state]*tierCount
    # selected_client_index = clientTier
    selected_client_data = [[federated_train_data[i] for i in tierSet] for tierSet in clientTier]
    for round_num in range(1, NUM_ROUNDS+1):
        flag=[0]*tierCount  # count the number of tiers involved
        weights=[len(clientTier[i]) for i in range(tierCount)]
        for tier in range(tierCount):
            # determine whether the current need to be calculated (multipliler and clients exist)
            if (round_num%(tier+1)==0 and weights[tier]!=0) :
                flag[tier] = 1
                # in there is some client in the tier, then do something
                states[tier], metrics = iteratives[tier].next(states[tier], selected_client_data[tier])
                flagSum=sum(flag)
                if flagSum==1:
                    globalState = states[tier]
                    if weights[tier] != 1:
                        for layer in range(len(globalState.model.trainable)):
                            globalState.model.trainable[layer]=(globalState.model.trainable[layer]*weights[tier])
                elif flagSum>1:
                    for layer in range(len(globalState.model.trainable)):
                        globalState.model.trainable[layer] = (globalState.model.trainable[layer]\
                                                             +states[tier].model.trainable[layer]*weights[tier])

        weightSum=np.sum(weights,where=flag)
        if weightSum>=2:
            for layer in range(len(globalState.model.trainable)):
                globalState.model.trainable[layer] = globalState.model.trainable[layer]/weightSum
            for tier in range(tierCount):
                if flag[tier]:
                    states[tier]=globalState
                #     for layer in range(len(globalState.model.trainable)):
                #         states[tier].model.trainable[layer] = globalState.model.trainable[layer]
                # #states[tier].model.trainable=globalState.model.trainable
        #print('Rep'+str(re)+'Agg'+str(tierCount)+'round {:2d}, {:2d} clients, Batch size {:2d}, {:2d} Epoch, metrics={}'.format(round_num,len(selected_client_index),batchSizeTrain,NUM_EPOCHS, list(metrics.items())[-2:]))
        print('Rep' + str(rept) + 'Agg' + str(tierCount) + 'round'+str(round_num))
        if round_num % TEST_frequency == 1:
            print(time.strftime("%Y%m%d %H%M%S"))
            metrics_valid = evaluation(globalState.model, federated_test_data)
            record_valid.append(metrics_valid['sparse_categorical_accuracy'])
            print('validation', metrics_valid)
    return record_valid

betaValues=[10] #2
tierTimes=[10] #1
repetition=3
f7validTier=np.zeros([repetition,len(tierTimes),len(betaValues),2,NUM_VALID])   # 2 for cs and aog
f7validNum=np.zeros([repetition,len(tierTimes),len(betaValues),NUM_VALID])
f7validNumTime=np.zeros([repetition,len(tierTimes),len(betaValues),NUM_ROUNDS])

for i in range(repetition):
    for j,tierTime in enumerate(tierTimes):
        for k,beta in enumerate(betaValues):
            print(i,j,k)
            x_train_noniid, y_train_noniid = nonIIDGen(beta=beta)
            federated_train_data = make_federated_data(x_train_noniid, y_train_noniid, 'train', seednumber=i)
            clientTier, latency = tierGenerator(tierTime)
            state = iterative.initialize()
            print('repetition' + str(i))
            # for k in range(tierShowMax):   # j start from 0, but +1 into the tier function
            #    print(repetition)
            #    valid5sTier[i,j,k,:]=fedAvgTier(state,k+1,i)
            f7validTier[i, j, k, 0, :] = fedAvgTier(state, i, 1)
            f7validTier[i, j, k, 1, :] = fedAvgTier(state, i, len(clientTier))
            f7validNum[i, j, k, :], f7validNumTime[i, j, k, :] = fedNum(state, clientCount)
            np.savez("mydata_f7_temp.npz",
                     f7validTier=f7validTier,
                     f7validNum=f7validNum,
                     f7validNumTime=f7validNumTime,

                     )

f7validTierAvg=np.average(f7validTier,axis=0)
f7validNumAvg=np.average(f7validNum,axis=0)
f7validNumTimeAvg=np.average(f7validNumTime,axis=0)



np.savez("mydata0120_f57.npz",

        f7validTierAvg=f7validTierAvg,
        f7validNumAvg=f7validNumAvg,
        f7validNumTimeAvg=f7validNumTimeAvg,
        f7validTier=f7validTier,
        f7validNum=f7validNum,
        f7validNumTime=f7validNumTime,

        )
print(1)