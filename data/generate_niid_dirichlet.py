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

clientCount=30
beta=1
data_len=600
train_len=500

# Setup directory for train/test data.pkl
train_path = './data.pkl/train/all_data_0_niid_0_keep_10_train_9.json'
test_path = './data.pkl/test/all_data_0_niid_0_keep_10_test_9.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get mldata data.pkl, normalize, and divide by level (std)
# mldata = fetch_mldata('mldata original', data_home='./data.pkl')

name='cifar-100'

try:
    with open(name+'.pkl', 'rb') as file:
        mlrawdata = pickle.load(file)
except:
    mlrawdata = fetch_openml('CIFAR-100', data_home='./data.pkl')
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(mlrawdata, f)

sampleCategories=len(np.unique(mlrawdata['target']))

mu = np.mean(mlrawdata.data.astype(np.float32), 0)
sigma = np.std(mlrawdata.data.astype(np.float32), 0)
mlrawdata.data = (mlrawdata.data.astype(np.float32) - mu) / (sigma + 0.0001)
mldata = []
for i in trange(sampleCategories):
    idx = mlrawdata.target == str(i)
    mldata.append(mlrawdata.data[idx])
print('Number of samples in each category:')
print([len(v) for v in mldata])


[np.random.shuffle(oneclass) for oneclass in mldata]
#mldata_data=[oneclass[0,] for oneclass in mldata_data]

# trim data.pkl

mldata_test=[oneclass[train_len:data_len] for oneclass in mldata]
mldata_train=[oneclass[000:train_len] for oneclass in mldata]


###### CREATE USER DATA SPLIT #######
# Assign 10 samples to each user, 10 samples belonging to 2 categories
X = [[] for _ in range(clientCount)]
y = [[] for _ in range(clientCount)]
for k in range(sampleCategories):
    proportions = np.random.dirichlet(np.repeat(beta, clientCount))
    #proportions = np.array([p * (len(idx_j) < 50000 / clientCount) for p, idx_j in zip(proportions, idx_batch)])
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(mldata_train[k])).astype(int)[:-1]
    X = [idx_j + idx.tolist() for idx_j, idx in zip(X, np.split(mldata_train[k], proportions))]
    y = [idx_j + [k]*len(idx) for idx_j, idx in zip(y, np.split(mldata_train[k], proportions))]


# Create data.pkl structure
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}

X_test=[i.tolist() for i in mldata_test]
y_test=reduce(lambda a,b: a+b,list(map(lambda x: [x]*(data_len-train_len),range(sampleCategories))))

for i in trange(clientCount):
    uname = 'f_{0:05d}'.format(i)
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])

    train_data['users'].append(uname)
    train_data['user_data'][uname] = {'x': X[i], 'y': y[i]}
    train_data['num_samples'].append(len(X[i]))
    # test_data['users'].append(uname)
    # test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    # test_data['num_samples'].append(test_len)
# Setup clientCount users


test_data['users'].append('Server')
test_data['user_data']['Server']={'x': X_test, 'y': y_test}
test_data['num_samples'].append(len(y_test))



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