## Latency Aware Semi-synchronous Client Selection and Model Aggregation for Wireless Federated Learning

![wfl.jpg](fig%2Fwfl.jpg)
### Abstract
Federated learning (FL) is a collaborative machine learning (ML) framework particularly suited for ML models requiring numerous training samples, such as Convolutional Neural Networks
(CNNs), Recurrent Neural Networks (RNNs), and Random Forest, in the context of various applications, e.g., next-word prediction and eHealth. FL involves various clients participating in the training
process by uploading their local models to an FL server in each global iteration. The server aggregates
these local models to update a global model. The traditional FL process may encounter bottlenecks,
known as the straggler problem, where slower clients delay the overall training time. This paper introduces the Latency awarE Semi-synchronous client Selection and mOdel aggregation for federated
learNing (LESSON) method. LESSON allows clients to participate at different frequencies: faster
clients contribute more frequently, thereby mitigating the straggler problem and expediting convergence. Moreover, LESSON provides a tunable trade-off between model accuracy and convergence
rate by setting varying deadlines. Simulation results show that LESSON outperforms two baseline
methods, namely FedAvg and FedCS, in terms of convergence speed and maintains higher model
accuracy as compared to FedCS.
### Semi-synchronous FL Scheduling (LESSON)
![LESSON_sch.jpg](fig%2FLESSON_sch.jpg)
### Non Independent and Identically Distribution
The simulation of the non-iid distribution across clients is conducted using a Dirichlet distribution characterized by a parameter β. A higher value of β results in a more uniform distribution. 

Rows represents different clients and columns indicates the portions of various data classes, differentiated by color.
![data_dis.jpg](fig%2Fdata_dis.jpg)
https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables
### Code
env: tensorflow-federated 

[main.py](main.py) is modified from https://www.tensorflow.org/federated/tutorials/building_your_own_federated_learning_algorithm

### Paper
https://www.mdpi.com/1999-5903/15/11/352
#### Citation
@Article{fi15110352,
AUTHOR = {Yu, Liangkun and Sun, Xiang and Albelaihi, Rana and Yi, Chen},
TITLE = {Latency-Aware Semi-Synchronous Client Selection and Model Aggregation for Wireless Federated Learning},
JOURNAL = {Future Internet},
VOLUME = {15},
YEAR = {2023},
NUMBER = {11},
ARTICLE-NUMBER = {352},
URL = {https://www.mdpi.com/1999-5903/15/11/352},
ISSN = {1999-5903},
DOI = {10.3390/fi15110352}
}