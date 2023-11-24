import numpy as np
import sys
import csv
import pandas as pd
from BasicFunctions import *

GoalMatrixRank = int(sys.argv[1]) # Set goal matrix rank
TITLE = sys.argv[2] # Set output file name
GoalVarianceNormalize = int(sys.argv[4])
#sth = int(sys.argv[4]) if len(sys.argv) >= 3 else -1 #Threshold of how long sample the network in early phase.

"""
Execute Evolutionary Simulation
"""

print("TITLE{}".format(TITLE))
print("Rank{}".format(GoalMatrixRank))


POPULATION_SIZE = 100
MAX_GENERATION = 50000#120000
nNode=6
nLayer=5
nMatrix = nLayer-1
MUTATION_RATE = 0.2/(nMatrix*nNode*nNode)


#### Simulation parameters #####

### Active node definition
### 0. Relative contribution of each node to the fitness.
### 1. Relative contribution of each node to the total in-out relation.
### 3. Relative strength of maximal interaction of each node.
ActiveNodeDefinition = 0

### GoalMatrixParameters
GoalMatrixRank = int(sys.argv[1])
GoalMatrixNorm = 60
GoalMatrixVariance = 1 if GoalVarianceNormalize else np.nan # If this set to np.nan, the variance is not normalized.
G_params = [GoalMatrixRank, GoalMatrixNorm, GoalMatrixVariance]

### Goal matrix chnage
GoalMatrixChange = False

### Goal matrix expansion
GoalExpansion = False
DuplicateTime = -1


### Initial interaction strength (A0) ###
INIT_NET_NORM = 0.01

Define_global_value_in_modules(nNode, nLayer, POPULATION_SIZE, MUTATION_RATE, None, ActiveNodeDefinition)


waist_std_list = list()
waist_mean_list = list()
waist_mode_list = list()
init_value = [0.01, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80]
for init_net_norm in init_value:
    waist_size_set = np.array(
        [np.min(GeneticAlgorithm(init_net_norm, G_params, MAX_GENERATION, output_style = 0)) for _ in range(100)]
        )
    waist_size_set = waist_size_set[~np.isnan(waist_size_set)]
    waist_std = np.std(waist_size_set)
    waist_mean = np.mean(waist_size_set)
    waist_mode = statistics_mode(waist_size_set)

    waist_std_list.append(waist_std)
    waist_mean_list.append(waist_mean)
    waist_mode_list.append(waist_mode)


result_dict = {'Initial_network_size': init_value, 'waist_std': waist_std_list, 'waist_mean': waist_mean_list, 'waist_mode': waist_mode_list}
result_df = pd.DataFrame.from_dict(result_dict, orient='index').T
result_df.to_csv('{}.csv'.format(TITLE))

print("RunID:{} Successfully Done!!".format(TITLE)) 