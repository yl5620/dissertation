# Dissertation  for the MSc in Statistics at Imperial College London
## Topic: Enhancing Dynamic Network Predictions with Node Representations and Temporal Point Processes

## Short Description

Here is a short description of what this project is about:
- Define the problem of dynamic graph link prediction under the temporal point process and attention based node embedding framework.
- Review the existing models, theorems, simulators and datasets that related to this task.
- Introduce improvements to the method to enhance model explanability, accuracy and stability. The improvements include: introduce multi-hop neighborhood scheme, decay factor and attribute factor and a new update rule to reduce the constraining.
- Introduce three simulators tailoring to the last three improvements.
- Conduct test on both synthetic dataset and real-world dataset.

This dissertation focuses on dynamic graphs where node relationships evolve, focusing on two main types of events: short-term communications (modeled by stochastic matrix \(S(t)\)) and long-term associations an (modeled by adjacency matrix \(A(t)\)) with the base model proposed by:
- Rakshit Trivedi (2019) https://openreview.net/pdf?id=HyePrhR5KX

This work reproduce the baseline model with reference to https://github.com/uoguelph-mlrg/LDG/tree/master, and enhance the model with the improvements mentioned above.

## Code Related
### Requirements
The code is based on Python3.11.7. There are few dependencies to run the code, and the libraries are listed below:
- numpy  1.26.1
- torch  2.1.0
- pandas 2.1.2

### Introduction of main code and usages
#### 1. Process and Load the data
Take loading the synthetic data an example:
```python
# import required packages
import numpy as np
import pandas as pd
import datetime
import torch
import torch.utils
from datetime import timezone

# import the loader
from Synthetic_with_attribute_data_loader import *

# load the processed data
train_set = SyntheticDataset('train', data_dir = ...)
test_set = SyntheticDataset('test',  data_dir = ...)

# set initialisations
initial_embeddings = np.random.randn (train_set.N_nodes, 32)
A_initial = train_set.get_Adjacency()
```

#### 2. Instantiate the model
```python
#import
from DyRep import *

model = DyRep_update(
node_embeddings=initial_embeddings,
A_initial=None,
N_surv_samples=5,
n_hidden=32,
node_degree_global= node_degree_global,
N_hops=2,
gamma = 0.5,
decay_rate = 0.0,
threshold=0.5,
with_attributes=False,
with_decay=False,
new_SA= False,
)
```
Parameters:
- node_embeddings: (np array) initial embedding of the nodes
- A_initial: (np array) initial adjacency matrix
- N_surv_samples: (int) number of events to sample
- n_hidden: (int) number of hidden features
- node_degree_global: (list) node degree
- N_hops: (int) number of hop of neighborhood considered in the model
- gamma: (float) weight of attribute 
- decay_rate: (float) factor in structure considering the decay of importance of events
- threshold: (float) boundary in new update rule
- with_attributes: (bool) consider attributes or not
- with_decay: (bool) consider decay or not
- new_SA: (bool) consider new update rule or not

### The code for training and evaluation could be found in ipynb file for each dataset
