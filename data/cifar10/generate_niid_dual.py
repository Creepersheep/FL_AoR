#from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os

clientCount=30
leastSamples=10
sampleCategories=10
categoriesOneclient=2

# Setup directory for train/test data.pkl
train_path = 'data/dual/train/all_data_0_niid_0_keep_10_train_9.json'
test_path = 'data/dual/test/all_data_0_niid_0_keep_10_test_9.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get cifar10 data.pkl, normalize, and divide by level (std)
# cifar10 = fetch_mldata('cifar10 original', data_home='./data.pkl')

cifar10 = fetch_openml('CIFAR_10', data_home='./data.pkl')
mu = np.mean(cifar10.data.astype(np.float32), 0)
sigma = np.std(cifar10.data.astype(np.float32), 0)
cifar10.data = (cifar10.data.astype(np.float32) - mu)/(sigma+0.001)
cifar10_data = []
for i in trange(sampleCategories):
    idx = cifar10.target == str(i)
    cifar10_data.append(cifar10.data[idx])
print('Number of samples in each category:')
print([len(v) for v in cifar10_data])


[np.random.shuffle(oneclass) for oneclass in cifar10_data]
cifar10_data=[np.moveaxis(np.reshape(oneclass, [-1,3, 32,32]),1,-1) for oneclass in cifar10_data]

# for oneclass in cifar10_data:
#     b=np.random.shuffle()

###### CREATE USER DATA SPLIT #######
# Assign 10 samples to each user, 10 samples belonging to 2 categories

X = [[] for _ in range(clientCount)]
y = [[] for _ in range(clientCount)]
idx = np.zeros(sampleCategories, dtype=np.int64)
for user in range(clientCount):
    for j in range(categoriesOneclient):
        l = (user+j)%sampleCategories
        lp=leastSamples//categoriesOneclient
        X[user] += cifar10_data[l][idx[l]:idx[l]+lp].tolist()
        y[user] += (l*np.ones(lp)).tolist()
        idx[l] += lp
print(idx)


# Assign remaining sample by power law
user = 0
# 0 mean, 2 sigma
#props = np.random.lognormal(0, 2.0, (10,int(clientCount/10),2))
props = np.random.lognormal(0, 2.0, (sampleCategories,int(clientCount/sampleCategories),categoriesOneclient))
props = np.array([[[len(v)-clientCount]] for v in cifar10_data])*props/np.sum(props,(1,2), keepdims=True)
#idx = 1000*np.ones(10, dtype=np.int64)
# for user in trange(clientCount):
#     for j in range(2):
#         l = (user+j)%10
#         num_samples = int(props[l,user//10,j])
#         #print(num_samples)
#         if idx[l] + num_samples < len(cifar10_data[l]):
#             X[user] += cifar10_data[l][idx[l]:idx[l]+num_samples].tolist()
#             y[user] += (l*np.ones(num_samples)).tolist()
#             idx[l] += num_samples
for user in trange(clientCount):
    for j in range(categoriesOneclient):
        l = (user+j)%sampleCategories
        num_samples = int(props[l,user//sampleCategories,j])
        #print(num_samples)
        if idx[l] + num_samples < len(cifar10_data[l]):
            X[user] += cifar10_data[l][idx[l]:idx[l]+num_samples].tolist()
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
