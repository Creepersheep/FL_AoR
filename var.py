import numpy as np
import matplotlib.pyplot as plt
# variables
# repetion=3
# NUM_VALID=2
# TEST_frequency=3


repetion=1
NUM_VALID=251
TEST_frequency=20

# repetion=1
# NUM_VALID=101
# TEST_frequency=10


NUM_ROUNDS = 1+TEST_frequency*(NUM_VALID-1)
VALID_RANGE=range(1,NUM_ROUNDS+1,TEST_frequency)
clientCount = 50
NUM_eva_clients=200   # eva_client must significantly larger than normal one then qualified clients could be selected.
NUM_EPOCHS = 1

SHUFFLE_BUFFER = 500
PREFETCH_BUFFER = 10

# partition
n_parties=clientCount

K = 10  # number of categories in cifar or mnist
NUM_SELECTED=5
# beta=10000   # diversity of data, lager value means IID, lower non-IID
betaValues=[0.1,1,10]
#betaValues=[0.01,0.1,1,10]
betaCount=len(betaValues)
batchSizeTrain = 20
batchCountTrain=1
# batchSizeTest = 10000
# batchCountTest=12
batchSizeTest = 1000
batchCountTest=10

lr=0.1
mm=0.0

beishu=1.5


# communication parameters
modelSize=100*10**3,  #... % bit size of trainning model
areaSize=1  #areaSize
fc=2*10^9
commAlpha=9.6
commBeta=0.28
pathLos=6
pathnLoss=26 #%pathloss_los',10^(pathloss_los/10),...%pathloss_nlos',10^(pathloss_nlos/10),...
#noise=2*10*10**(-104/10-3)#   % -104dbm %env_noise',10^((-174-30)/10)*15*10^3, noise watt/1MHz
#bandWidth=100*10**3 #);%bandwidth_max',20*2^50/10^10,... %20 MHz, 10^10 is used to decrease the size of the number
commPower=0.1  # %100 mW over all bandwidth
noise=10**(-100/10-3)
bandWidth=30*10**3


# computation parameters
compPower=3#,... %upload power was 1
sampleCount=batchSizeTrain*batchCountTrain  #D trainning samples
gamma=10**(-28)#,... %switch gap
epsilon=5*10**(-2)#,... model calculation resolution
theta=1#,...
# a=100#);

# tier parameters
tierTimes=[20,30,40]
tierTime=tierTimes[0]
tierTimesCount=len(tierTimes)
tierShowMax=3



def latencyGenerator():
    location = np.random.rand(clientCount, 2) * (areaSize * 2) - areaSize   #########################
    distance = np.sqrt(location[:, 0] ** 2 + location[:, 1] ** 2)
    pathLos = 128.1 + 37.6 * np.log10(distance)
    commRate = bandWidth * np.log2(1 + commPower * np.power(10, -pathLos / 10) / noise)
    commLatency = modelSize / commRate

    # computation random generator
    cycleBit = np.random.uniform(3, 5, clientCount) * 10 ** 8   ####################################
    compFrequency = np.random.uniform(0.8, 3, clientCount) * 10 ** 9
    compLatency = theta * np.log2(1 / epsilon) * cycleBit * sampleCount / compFrequency
    latency = commLatency + compLatency
    return latency

def tierGenerator(tierTime=tierTimes[1]):
# distribution over 30: 3.01790,8.09620,5.80710,4.28270,3.25280,1.93300,1.22620,0.83030,0.58600,0.41520
    # communication random generator
    latency=latencyGenerator()
    # number of tiers to be considers
    #clientTier = [np.where(np.logical_and(tierTime * (i - 1) < latency,
    #                                      latency < tierTime * (i)))[0]
    #              for i in range(1, tierCountMax + 1)]
    clientTier=[]
    counted=0
    i=1
    while counted<clientCount:
        temp=np.where(np.logical_and(tierTime * (i - 1) < latency,
                                          latency < tierTime * (i)))[0]
        clientTier.append(temp)
        counted=counted+len(temp)
        i=i+1

    # gurantee at least one client in iter 1, not necessary
    if len(clientTier[0])==0:
        min_index=np.argmin(latency)
        clientTier[0]=np.append(clientTier[0],min_index)
        minValTier=int(np.floor(latency[min_index]/tierTime))
        clientTier[minValTier]=np.delete(clientTier[minValTier],np.where(clientTier[minValTier]==min_index))
    return clientTier,latency


def tierGenerator_latency(latency,tierTime=tierTimes[1]):
# distribution over 30: 3.01790,8.09620,5.80710,4.28270,3.25280,1.93300,1.22620,0.83030,0.58600,0.41520
    # communication random generator
    # latency=latencyGenerator()
    # number of tiers to be considers
    #clientTier = [np.where(np.logical_and(tierTime * (i - 1) < latency,
    #                                      latency < tierTime * (i)))[0]
    #              for i in range(1, tierCountMax + 1)]
    clientTier=[]
    counted=0
    i=1
    while counted<clientCount:
        temp=np.where(np.logical_and(tierTime * (i - 1) < latency,
                                          latency < tierTime * (i)))[0]
        clientTier.append(temp)
        counted=counted+len(temp)
        i=i+1

    # gurantee at least one client in iter 1, not necessary
    if len(clientTier[0])==0:
        min_index=np.argmin(latency)
        clientTier[0]=np.append(clientTier[0],min_index)
        minValTier=int(np.floor(latency[min_index]/tierTime))
        clientTier[minValTier]=np.delete(clientTier[minValTier],np.where(clientTier[minValTier]==min_index))
    return clientTier

avemaxlatency=[]
for i in range(50):
    clientTier, latency = tierGenerator()
    avemaxlatency.append(max(latency))
    print(max(latency))
fedavglatency=np.average(avemaxlatency)
clientTier,latency=tierGenerator()
plt.hist(latency)
plt.show()
#clientTier,latency,clientTier_asyn,clientTier_latency_asyn=tierGenerator()


print(1)