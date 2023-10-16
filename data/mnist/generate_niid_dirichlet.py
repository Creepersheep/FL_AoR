#from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os
import pickle
import json
from functools import reduce
import itertools
clientCount=30
betas=[0.1,1,10]

# sample count for each categories
data_len=7000
train_len=6000
# data_len=15
# train_len=10


name='mnist_784'

try:
    with open(name+'.pkl', 'rb') as file:
        mlrawdata = pickle.load(file)
except:
    mlrawdata = fetch_openml('mnist_784', data_home='./data.pkl')
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(mlrawdata, f)

sampleCategories=len(np.unique(mlrawdata['target']))

mu = np.mean(mlrawdata.data.astype(np.float32), 0)
sigma = np.std(mlrawdata.data.astype(np.float32), 0)
mlrawdata.data = (mlrawdata.data.astype(np.float32) - mu) / (sigma + 0.0001)

for beta in betas:
    # Setup directory for train/test data.pkl
    train_path = './data/dirichlet'+str(beta)+'/train/all_data_0_niid_0_keep_10_train_9.json'
    test_path = './data/dirichlet'+str(beta)+'/test/all_data_0_niid_0_keep_10_test_9.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    mldata = []


    # for training
    for i in trange(sampleCategories):
        idx = mlrawdata.target[:60000] == str(i)
        idx=np.concatenate((idx,np.full([10000],False)))
        mldata.append(mlrawdata.data[idx])
    print('Number of samples in each category:')
    print([len(v) for v in mldata])


    [np.random.shuffle(oneclass) for oneclass in mldata]
    # mldata=[np.reshape(oneclass,[-1,3,32,32]) for oneclass in mldata]
    # mldata_data=[oneclass[0,] for oneclass in mldata_data]

    # trim data.pkl

   # mldata_test=[oneclass[train_len:data_len] for oneclass in mldata]
    #mldata_train=[oneclass[000:train_len] for oneclass in mldata]
    mldata_train=mldata


    ###### CREATE USER DATA SPLIT #######
    # Assign 10 samples to each user, 10 samples belonging to 2 categories
    X = [[] for _ in range(clientCount)]
    y = [[] for _ in range(clientCount)]
    for k in range(sampleCategories):
        proportions = np.random.dirichlet(np.repeat(beta, clientCount))
        #proportions = np.array([p * (len(idx_j) < 50000 / clientCount) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(mldata_train[k])).astype(int)[:-1]
        X = [X + idx.tolist() for X, idx in zip(X, np.split(mldata_train[k], proportions))]
        y = [y + [k]*len(idx) for y, idx in zip(y, np.split(mldata_train[k], proportions))]


    # Create data.pkl structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # X_test=[i for i in mldata_test]
    # X_test=list(itertools.chain(*mldata_test))
    # y_test=reduce(lambda a,b: a+b,list(map(lambda x: [x]*(data_len-train_len),range(sampleCategories))))
    X_test=mlrawdata.data[60000:]
    y_test=mlrawdata.target[60000:]


    # fit the original
    ave=len(X_test)/clientCount
    # test_data['users'].append('Server')
    # test_data['user_data']['Server']={'x': X_test, 'y': y_test}
    # test_data['num_samples'].append(len(y_test))
    testprop=(np.linspace(0,1,num=clientCount,endpoint=False)*len(X_test)).astype(int)[1:]
    X_test_c = [[] for _ in range(clientCount)]
    y_test_c = [[] for _ in range(clientCount)]
    X_test_c = [X + idx.tolist() for X, idx in zip(X_test_c, np.split(X_test, testprop))]
    y_test_c = [y + idx.tolist() for y, idx in zip(y_test_c, np.split(y_test, testprop))]

    for i in trange(clientCount):
        uname = 'f_{0:05d}'.format(i)
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i], 'y': y[i]}
        train_data['num_samples'].append(len(X[i]))
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_test_c[i], 'y': y_test_c[i]}
        test_data['num_samples'].append(len(X_test_c[i]))
    # Setup clientCount users






    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)
    print('finished')

    # if K == 2 and n_parties <= 10:
        #     if np.min(proportions) < 200:
        #         min_size = 0
        #         break

    #     for j in range(clientCount):
    #         np.random.shuffle(idx_batch[j])
    #         net_dataidx_map[j] = idx_batch[j]
    #
    # traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    # return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)