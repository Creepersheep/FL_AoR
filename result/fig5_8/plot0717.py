import itertools
import math
import pickle

import numpy as np
from matplotlib import pyplot as plt

# open a file, where you stored the pickled data
import var

grid = []
model = 'cifar10'
methods_with_repetion = ['aog']
methods_without_repetion = ['avg']
utiers = [0, 1, 2, 3]
repetion = 3

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
    'tierTimes': var.tierTimes,
    'utiers': [0, 1],
    'methods': methods_with_repetion,
    'model': [model]
})

# fedavg on non-IID
grid.append({
    'betaValues': [0.1, 1, 10],
    'methods': methods_without_repetion,
    'model': [model]
})

setting_list = []
for i, onegrid in enumerate(grid):
    setting_list += [dict(zip(onegrid.keys(), values)) for values in itertools.product(*onegrid.values())]
setting_list1 = setting_list
for i, d1 in reversed(list(enumerate(setting_list))):
    for j, d2 in enumerate(setting_list[0:i - 1]):
        if i != 0 and d1 == d2:
            setting_list.pop(i)

data = []
data_sl = []

# for i in range(len(setting_list)):
#     file = open('3kforall_9_'+str(i)+'.pkl', 'rb')
#     data=data+[pickle.load(file)]
#     print(len(data))
#     data_sl=data_sl+[pickle.load(file)]

file = open('3kforall_9.pkl', 'rb')
data = pickle.load(file)
# print(len(data))
data_sl = pickle.load(file)

file = open('mnist0728.pkl', 'rb')
data = data + pickle.load(file)
# print(len(data))
data_sl = data_sl + pickle.load(file)

# newdata=[]
# newlist=[]
# for i, oneprintset in enumerate(setting_list):
#     if 'repetion' in setting_list[i].keys():
#         if oneprintset['repetion'] == 0:
#             same=[]
#             for t in range(var.repetion):
#                 oneprintset['repetion']=t
#                 for j, correset in enumerate(setting_list):
#                     if correset==oneprintset:
#                         print(j)
#                         same.append(data[j])
#             oneprintset.pop('repetion')
#             newdata.append(np.average(same, axis=0))
#             newlist.append(oneprintset)
#     else:
#         for j, correset in enumerate(setting_list):
#             if correset == oneprintset:
#                 newdata.append(data[j])
#                 newlist.append(oneprintset)
#
# print(1)


sizeH = 9
sizeV = 2.5
VALID_RANGE = var.VALID_RANGE
NUM_VALID = var.NUM_VALID
TEST_frequency = var.TEST_frequency
fedavglatency = var.fedavglatency
betaValues = [0.1, 1, 10]

# figure 4 mnist and cifar
mnist_range = 201
mnist_valid_range = range(1, 2002, 10)
tierTimes = var.tierTimes
fig4, axs = plt.subplots(ncols=4, sharey=True, figsize=[sizeH, sizeV], tight_layout=True)
axs[0].plot(VALID_RANGE,
            data[data_sl.index(dict(betaValues=1, tierTimes=20, methods='aog', utiers=0, model='cifar10'))],
            label='LESSON')
axs[0].plot(VALID_RANGE,
            data[data_sl.index(dict(betaValues=1, tierTimes=20, methods='aog', utiers=1, model='cifar10'))],
            label='FedCS')
axs[0].plot(VALID_RANGE, data[data_sl.index(dict(betaValues=1, methods='avg', model='cifar10'))],
            label='FedAvg')  # beta=1
axs[0].set(xlabel='Iterations \n(a) CIFAR-10', ylabel='Test accuracy')

# axs[1].plot(VALID_RANGE[0:mnist_range], validNumAvg1_mnist[1,0:mnist_range],label='FedAvg') #1 refers to beat=1
# axs[1].plot(VALID_RANGE[0:mnist_range], valid5sTierAvg_mnist[1, 0, 0:mnist_range], label='FedCS')
# axs[1].plot(VALID_RANGE[0:mnist_range], valid5sTierAvg_mnist[1, 1, 0:mnist_range], label='LESSON')
axs[1].plot(mnist_valid_range,
            data[data_sl.index(dict(betaValues=1, tierTimes=20, methods='aog', utiers=0, model='mnist'))][:mnist_range],
            label='LESSON')
axs[1].plot(mnist_valid_range,
            data[data_sl.index(dict(betaValues=1, tierTimes=20, methods='aog', utiers=1, model='mnist'))][:mnist_range],
            label='FedCS')
axs[1].plot(mnist_valid_range, data[data_sl.index(dict(betaValues=1, methods='avg', model='mnist'))][:mnist_range],
            label='FedAvg')
axs[1].set(xlabel='Iterations\n(b) MNIST', ylabel='Test accuracy')

# f5fit = math.ceil(NUM_VALID*(deadline / fedavglatency))
# f4fit_cifar=math.ceil(20/fedavglatency*var.NUM_VALID)
f4fit_cifar = math.ceil(NUM_VALID * (20 / fedavglatency))

axs[2].plot(np.arange(NUM_VALID) * 20 * TEST_frequency,
            data[data_sl.index(dict(betaValues=1, tierTimes=20, methods='aog', utiers=0, model='cifar10'))],
            label='LESSON')
axs[2].plot(np.arange(NUM_VALID) * 20 * TEST_frequency,
            data[data_sl.index(dict(betaValues=1, tierTimes=20, methods='aog', utiers=1, model='cifar10'))],
            label='FedCS')
axs[2].plot(np.arange(f4fit_cifar) * fedavglatency * TEST_frequency,
            data[data_sl.index(dict(betaValues=1, methods='avg', model='cifar10'))][:f4fit_cifar], label='FedAvg')
axs[2].set(xlabel='Time (s)\n(c) CIFAR-10', ylabel='Test accuracy')
tick_positions = np.arange(0, 100002, 50000)
tick_labels = [f"{int(val / 1000)},000" for val in tick_positions]
tick_labels[0] = 0
axs[2].set_xticks(tick_positions, tick_labels)
# axs[2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

f4fit_mnist = math.ceil(20 / fedavglatency * mnist_range)
# axs[3].plot(np.arange(f4fit_mnist)*validNumTimeAvg1_mnist[1,0]*TEST_frequency,validNumAvg1_mnist[1,0:f4fit_mnist], label='FedAvg')
# axs[3].plot(np.arange(mnist_range)*tierTimes[1]*TEST_frequency, valid5sTierAvg_mnist[1, 0, 0:mnist_range], label='FedCS')
# axs[3].plot(np.arange(mnist_range)*tierTimes[1]*TEST_frequency, valid5sTierAvg_mnist[1, 1, 0:mnist_range],  label='LESSON')
axs[3].plot(np.arange(mnist_range) * 20 * TEST_frequency,
            data[data_sl.index(dict(betaValues=1, tierTimes=20, methods='aog', utiers=0, model='mnist'))][:mnist_range],
            label='LESSON')
axs[3].plot(np.arange(mnist_range) * 20 * TEST_frequency,
            data[data_sl.index(dict(betaValues=1, tierTimes=20, methods='aog', utiers=1, model='mnist'))][:mnist_range],
            label='FedCS')
axs[3].plot(np.arange(f4fit_mnist) * fedavglatency * TEST_frequency,
            data[data_sl.index(dict(betaValues=1, methods='avg', model='mnist'))][:f4fit_mnist], label='FedAvg')
axs[3].set(xlabel='Time (s)\n(d) MNIST', ylabel='Test accuracy')
tick_positions = np.arange(0, 80001, 40000)
tick_labels = [f"{int(val / 1000)},000" for val in tick_positions]
tick_labels[0] = 0
axs[3].set_xticks(tick_positions, tick_labels)

# axs[3].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
for i in range(4):
    axs[i].grid()
axs[0].legend()
fig4.show()
fig4.savefig('f4.pdf')

## figure 5 different methods with different betas by iterations
fig5, axs = plt.subplots(ncols=3, sharey=True, figsize=[sizeH, sizeV], tight_layout=True)
for k, beta in enumerate(betaValues):
    # for j,deadline in enumerate(tierTimes[1:2]):
    deadline = var.tierTimes[0]
    axs[k].plot(VALID_RANGE, data[
        data_sl.index(dict(betaValues=beta, tierTimes=deadline, methods='aog', utiers=0, model='cifar10'))],
                label='LESSON')
    axs[k].plot(VALID_RANGE, data[
        data_sl.index(dict(betaValues=beta, tierTimes=deadline, methods='aog', utiers=1, model='cifar10'))],
                label='FedCS')
    axs[k].plot(VALID_RANGE, data[data_sl.index(dict(betaValues=beta, methods='avg', model='cifar10'))], label='FedAvg')
    axs[k].grid()
    axs[k].set(xlabel='Iterations\n(' + chr(97 + k) + ') \u03B2=' + str(beta), ylabel='Test accuracy')
axs[0].legend()
fig5.savefig('f50.pdf')
fig5.savefig('f50.jpg')
fig5.show()

# fig5, axs = plt.subplots(ncols=3,sharey=True,figsize=[sizeH,sizeV],tight_layout=True)
# for k,beta in enumerate(betaValues):
#     #for j,deadline in enumerate(tierTimes[1:2]):
#     deadline=var.tierTimes[1]
#     axs[k].plot(VALID_RANGE, data[data_sl.index(dict(betaValues=beta,tierTimes=deadline,methods='aog',utiers=0,model='cifar10'))], label='LESSON')
#     axs[k].plot(VALID_RANGE, data[data_sl.index(dict(betaValues=beta,tierTimes=deadline,methods='aog',utiers=1,model='cifar10'))], label='FedCS')
#     axs[k].plot(VALID_RANGE, data[data_sl.index(dict(betaValues=beta,methods='avg',model='cifar10'))], label='FedAvg')
#     axs[k].grid()
#     axs[k].set(xlabel='Iterations\n(' + chr(97 + k) + ') \u03B2=' + str(beta), ylabel='Test accuracy')
# axs[0].legend()
# fig5.savefig('f51.pdf')
# fig5.savefig('f51.jpg')
# fig5.show()


## figure 5_1 different methods with different betas by time
fig5_1, axs = plt.subplots(ncols=3, sharey=True, figsize=[sizeH, sizeV], tight_layout=True)
tick_positions = np.arange(0, 100001, 50000)
tick_labels = [f"{int(val / 1000)},000" for val in tick_positions]
tick_labels[0] = 0
for k, beta in enumerate(betaValues):
    for j in range(0, 1):
        deadline = var.tierTimes[j]
        f5fit = math.ceil(NUM_VALID * (deadline / fedavglatency))
        # axs[k].plot(VALID_RANGE, validTierAvg[j, k, 1, :], label='LESSON')
        # axs[k].plot(VALID_RANGE, validTierAvg[j, k, 0, :], label='FedCS')
        # axs[k].plot(VALID_RANGE, validNumAvg[j, k, :], label='FedAvg')
        axs[k].plot(np.arange(NUM_VALID) * deadline * TEST_frequency, data[
            data_sl.index(dict(betaValues=beta, tierTimes=deadline, methods='aog', utiers=0, model='cifar10'))],
                    label='LESSON')
        axs[k].plot(np.arange(NUM_VALID) * deadline * TEST_frequency, data[
            data_sl.index(dict(betaValues=beta, tierTimes=deadline, methods='aog', utiers=1, model='cifar10'))],
                    label='FedCS')
        axs[k].plot(np.arange(f5fit) * fedavglatency * TEST_frequency,
                    data[data_sl.index(dict(betaValues=beta, methods='avg', model='cifar10'))][:f5fit], label='FedAvg')
        axs[k].set_xticks(tick_positions, tick_labels)
    axs[k].grid()
    axs[k].set(xlabel='Time (s)\n(' + chr(97 + k) + ') \u03B2=' + str(beta), ylabel='Test accuracy')
axs[0].legend()

fig5_1.savefig('f5_1.pdf')
fig5_1.savefig('f5_1.jpg')
fig5_1.show()

## figure 7 compare fedaog with different tierTimes by iterations
fig7, axs = plt.subplots(ncols=2, sharey=True, figsize=[sizeH, sizeV], tight_layout=True)
for k, beta in enumerate(betaValues[0:2]):
    for j, deadline in enumerate(var.tierTimes):
        axs[k].plot(VALID_RANGE,
                    data[setting_list.index(
                        dict(betaValues=beta, tierTimes=deadline, utiers=0, methods='aog', model='cifar10'))],
                    label='\u03C4=' + str(deadline) + 's')
    axs[k].grid()
    axs[k].set(xlabel='Iterations\n(' + chr(97 + k) + ') \u03B2=' + str(beta), ylabel='Test accuracy')
axs[0].legend()
fig7.savefig('f7.pdf')
fig7.savefig('f7.jpg')
fig7.show()

## figure 7_1 compare fedaog with different tierTimes by time
fig7_1, axs = plt.subplots(ncols=2, sharey=True, figsize=[sizeH, sizeV], tight_layout=True)
for k, beta in enumerate(betaValues[0:2]):
    for j, deadline in enumerate(var.tierTimes):
        # f7fig=math.ceil(NUM_VALID/2**(j))
        f7fit = math.ceil(NUM_VALID * (var.tierTimes[0] / deadline))
        axs[k].plot(np.arange(f7fit) * deadline * TEST_frequency,
                    data[setting_list.index(
                        dict(betaValues=beta, tierTimes=deadline, utiers=0, methods='aog', model='cifar10'))][:f7fit],
                    label='\u03C4=' + str(deadline) + 's')
        axs[k].grid()
        axs[k].set(xlabel='Time (s)\n(' + chr(97 + k) + ') \u03B2=' + str(beta), ylabel='Test accuracy')
axs[0].legend()
fig7_1.savefig('f7_1.pdf')
fig7_1.savefig('f7_1.jpg')
fig7_1.show()
