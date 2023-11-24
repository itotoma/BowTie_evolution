import numpy as np
import sys
import csv
import pandas as pd
from BasicFunctions import *

GoalMatrixRank = int(sys.argv[1]) # Set goal matrix rank
TITLE = sys.argv[2] # Set output file name
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
GoalMatrixVariance = np.nan # If this set to np.nan, the variance is not normalized.
G_params = [GoalMatrixRank, GoalMatrixNorm, GoalMatrixVariance]

### Goal matrix chnage
GoalMatrixChange = False

### Goal matrix expansion
GoalExpansion = False
DuplicateTime = -1

### Initial interaction strength (A0) ###
INIT_NET_NORM = 0.01

Define_global_value_in_modules(nNode, nLayer, POPULATION_SIZE, MUTATION_RATE, None, ActiveNodeDefinition)
active_node_set = list()

for _ in range(100):
    active_node = GeneticAlgorithm(INIT_NET_NORM, G_params, MAX_GENERATION, output_style = 0)
    active_node_set.append(active_node)

active_node_data = np.array(active_node_set)
filename = TITLE + ".txt"
np.savetxt(filename, active_node_data)


