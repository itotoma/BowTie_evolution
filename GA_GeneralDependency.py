import numpy as np
import sys
import csv
import pandas as pd
from BasicFunctions import *

GoalMatrixRank = int(sys.argv[1]) # Set goal matrix rank
TITLE = sys.argv[2] # Set output file name
sth = int(sys.argv[3]) #Threshold of how long sample the network in early phase.
GoalVarianceNormalize = int(sys.argv[4])
#e.g., python GA_Dependnecy 1 'test' 50000 


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

Define_global_value_in_modules(nNode, nLayer, POPULATION_SIZE, MUTATION_RATE, None, ActiveNodeDefinition)

END_NETWORK = list()
INITIAL_NETWORKS = list()
GOAL_MATRICES = list()


init_value = [0.01, 0.1]#, 1.0, 2.0, 4.0, 6.0, 8.0, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80]
for init_net_norm in init_value:
    endnetwork = list()
    initialnetwork = list()
    goalmatrix = list()
    for _ in range(2):
        steadystate_network, initialstate_networks, goal_matrix = GeneticAlgorithm(init_net_norm, G_params, MAX_GENERATION, output_style=2, sth=sth)
        endnetwork.append(steadystate_network)
        initialnetwork.append(initialstate_networks)
        goalmatrix.append(goal_matrix)

    END_NETWORK.append(endnetwork)
    INITIAL_NETWORKS.append(initialnetwork)
    GOAL_MATRICES.append(goalmatrix)

END_NETWORK = np.array(END_NETWORK, dtype=object)
np.save(f"{TITLE}_END", END_NETWORK)

GOAL_MATRICES = np.array(GOAL_MATRICES, dtype=object)
np.save(f"{TITLE}_GOAL", GOAL_MATRICES)

INITIAL_NETWORKS = np.array(INITIAL_NETWORKS, dtype=object)
np.save(f"{TITLE}_INIT", INITIAL_NETWORKS)


print("RunID:{} Successfully Done!!".format(TITLE))


