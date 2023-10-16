import numpy as np
import math
from matplotlib import pyplot as plt
import pickle
# open a file, where you stored the pickled data
import var
import itertools
import copy



grid = []
model='cifar10'
methods_with_repetion = ['aog']
methods_without_repetion = ['avg']
utiers = [0, 1, 2, 3]
repetion=3
tierTimes=[10,20,60]
# imact of tier time
grid.append({
    'betaValues': [0.1, 1],
    'tierTimes':tierTimes,
    'utiers': [0],
    'methods': methods_with_repetion,
    'model': [model]
})

setting_list=[]
for i, onegrid in enumerate(grid):
    setting_list += [dict(zip(onegrid.keys(),values)) for values in itertools.product(*onegrid.values())]
setting_list1=setting_list
for i, d1 in reversed(list(enumerate(setting_list))):
    for j, d2 in enumerate(setting_list[0:i-1]):
        if i!=0 and d1==d2:
            setting_list.pop(i)
#
# data=[]
# data_sl=[]
#
# for i in range(len(setting_list)):
#     file = open('3kforall_'+str(i)+'.pkl', 'rb')
#     data=data+[pickle.load(file)]
#     print(len(data))
#     data_sl=data_sl+[pickle.load(file)]

file = open('3kforall.pkl', 'rb')
data=pickle.load(file)
# print(len(data))
data_sl=pickle.load(file)

newdata=[]
for i, onelist in enumerate(setting_list):
    dindex = [index for (index, item) in enumerate(data_sl) if
       item == onelist]
    samelist=[data[j] for j in dindex]
    newdata.append(np.average(samelist,0))

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


sizeH=9
sizeV=2.5
VALID_RANGE=var.VALID_RANGE
NUM_VALID=var.NUM_VALID
TEST_frequency=var.TEST_frequency
fedavglatency=var.fedavglatency
betaValues=[0.1,1,10]

## figure 7 compare fedaog with different tierTimes by iterations
fig7, axs = plt.subplots(ncols=2,sharey=True,figsize=[sizeH,sizeV],tight_layout=True)
for k,beta in enumerate(betaValues[0:2]):
    for j,deadline in enumerate(tierTimes):
        axs[k].plot(VALID_RANGE,
                    newdata[setting_list.index(dict(betaValues=beta,tierTimes=deadline,utiers=0,methods='aog',model='cifar10'))],
                    label='\u03C4='+str(deadline)+'s')
    axs[k].grid()
    axs[k].set(xlabel='Iterations\n(' + chr(97 + k) + ') \u03B2=' + str(beta), ylabel='Test accuracy')
axs[0].legend()
fig7.savefig('f7.pdf')
fig7.savefig('f7.jpg')
fig7.show()

## figure 7_1 compare fedaog with different tierTimes by time
fig7_1, axs = plt.subplots(ncols=2,sharey=True,figsize=[sizeH,sizeV],tight_layout=True)
for k,beta in enumerate(betaValues[0:2]):
    for j,deadline in enumerate(tierTimes):
        #f7fig=math.ceil(NUM_VALID/2**(j))
        f7fit = math.ceil(NUM_VALID*(tierTimes[0] / deadline))
        axs[k].plot(np.arange(f7fit)*deadline*TEST_frequency,
                    newdata[setting_list.index(dict(betaValues=beta,tierTimes=deadline,utiers=0,methods='aog',model='cifar10'))][:f7fit],
                    label='\u03C4='+str(deadline)+'s')
        axs[k].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        axs[k].grid()
        axs[k].set(xlabel='Time (s)\n(' + chr(97 + k) + ') \u03B2=' + str(beta), ylabel='Test accuracy')
axs[0].legend()
fig7_1.savefig('f7_1.pdf')
fig7_1.savefig('f7_1.jpg')
fig7_1.show()