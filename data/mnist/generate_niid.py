#from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os

clientCount=30

# Setup directory for train/test data.pkl
train_path = './data/train/all_data_0_niid_0_keep_10_train_9.json'
test_path = './data/test/all_data_0_niid_0_keep_10_test_9.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get MNIST data.pkl, normalize, and divide by level (std)
# mnist = fetch_mldata('MNIST original', data_home='./data.pkl')

# data_path='./data.pkl/openml'
# if not os.path.exists(data_path):
mnist = fetch_openml('mnist_784', data_home='./data.pkl')
mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)
mnist_data = []
for i in trange(10):
    idx = mnist.target == str(i)
    mnist_data.append(mnist.data[idx])
print('Number of samples in each category:')
print([len(v) for v in mnist_data])


[np.random.shuffle(oneclass) for oneclass in mnist_data]
mnist_data=[oneclass[0:1000] for oneclass in mnist_data]

# for oneclass in mnist_data:
#     b=np.random.shuffle()

###### CREATE USER DATA SPLIT #######
# Assign 10 samples to each user, 10 samples belonging to 2 categories
X = [[] for _ in range(clientCount)]
y = [[] for _ in range(clientCount)]
idx = np.zeros(10, dtype=np.int64)
for user in range(clientCount):
    for j in range(2):
        l = (user+j)%10
        X[user] += mnist_data[l][idx[l]:idx[l]+5].tolist()
        y[user] += (l*np.ones(5)).tolist()
        idx[l] += 5
print(idx)

# Assign remaining sample by power law
user = 0
# 0 mean, 2 sigma
props = np.random.lognormal(0, 2.0, (10,int(clientCount/10),2))
props = np.array([[[len(v)-clientCount]] for v in mnist_data])*props/np.sum(props,(1,2), keepdims=True)
#idx = 1000*np.ones(10, dtype=np.int64)
for user in trange(clientCount):
    for j in range(2):
        l = (user+j)%10
        num_samples = int(props[l,user//10,j])
        #print(num_samples)
        if idx[l] + num_samples < len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:idx[l]+num_samples].tolist()
            y[user] += (l*np.ones(num_samples)).tolist()
            idx[l] += num_samples

print(idx)

# Create data.pkl structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

# Setup clientCount users
for i in trange(clientCount):
    uname = 'f_{0:05d}'.format(i)
    
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.8*num_samples)
    test_len = num_samples - train_len
    
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print(train_data['num_samples'])
print(sum(train_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)
