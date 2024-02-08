## Latency Aware Semi-synchronous Client Selection and Model Aggregation for Wireless Federated Learning

![wfl.jpg](fig%2Fwfl.jpg)
### Abstract
Federated learning (FL) is a collaborative machine learning (ML) framework particularly 1
suited for ML models requiring numerous training samples, such as Convolutional Neural Networks 2
(CNNs), Recurrent Neural Networks (RNNs), and Random Forest, in the context of various applica- 3
tions, e.g., next-word prediction and eHealth. FL involves various clients participating in the training 4
process by uploading their local models to an FL server in each global iteration. The server aggregates 5
these local models to update a global model. The traditional FL process may encounter bottlenecks, 6
known as the straggler problem, where slower clients delay the overall training time. This paper in- 7
troduces the Latency awarE Semi-synchronous client Selection and mOdel aggregation for federated 8
learNing (LESSON) method. LESSON allows clients to participate at different frequencies: faster 9
clients contribute more frequently, thereby mitigating the straggler problem and expediting conver- 10
gence. Moreover, LESSON provides a tunable trade-off between model accuracy and convergence 11
rate by setting varying deadlines. Simulation results show that LESSON outperforms two baseline 12
methods, namely FedAvg and FedCS, in terms of convergence speed and maintains higher model 13
accuracy as compared to FedCS.
### Semi-synchronous FL Scheduling (LESSON)
![LESSON_sch.jpg](fig%2FLESSON_sch.jpg)
### Non Independent and Identically Distribution
The non-iid sample distribution on clients is simulated with a Dirichlet distribution with β. Larger β makes the distribution more evenly.
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