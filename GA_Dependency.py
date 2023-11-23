from scipy.stats import special_ortho_group
import numpy as np
import random
import copy
import math
import sys
import os 
from BasicFunctions import *

print("Start with {}".format(sys.argv[1]))
GoalMatrixRank = int(sys.argv[1]) # Set goal matrix rank
TITLE = sys.argv[2] # Set output file name
sth = int(sys.argv[3]) #Threshold of how long sample the network in early phase.

#e.g., python GA_Dependnecy 1 'test' 50000 


def GeneticAlgorithm(norm, GoalMatrixRank, sth, DesiredGoal):

    current_generation_individual_group_pre = [CreateInd(nNode,nMatrix,norm) for i in range(POPULATION_SIZE)]
    current_generation_individual_group = [set_eva(Ind, DesiredGoal) for Ind in current_generation_individual_group_pre]
    initial_network_size = np.mean([np.linalg.norm(Total_in_out(Ind.getGenom()), 'fro') for Ind in current_generation_individual_group])
    
    PreGenFitness = 0

    TIME_SERIES_NETWORK = list()
    for count in range(MAX_GENERATION):

        next_generation_network = [copy.deepcopy(Ind.getGenom()) for Ind in current_generation_individual_group]
        next_generation_individual_group = [genom(Network, 0) for Network in next_generation_network]
        next_generation_individual_group_mutated = [add_mutation(Ind, MUTATION_RATE) for Ind in next_generation_individual_group]
        next_generation_individual_group_evaluated = [set_eva(Ind, DesiredGoal) for Ind in  next_generation_individual_group_mutated]
        #Selection
        mixed_population = current_generation_individual_group + next_generation_individual_group_evaluated
        #tournament size is 4
        selected_group = select(mixed_population, 4)
        #Generation change
        selected_net = [copy.deepcopy(Ind.getGenom()) for Ind in selected_group]
        selected_eva = [Ind.getEvaluation() for Ind in selected_group]
        current_generation_individual_group = [genom(selected_net[i], selected_eva[i]) for i in range(POPULATION_SIZE)]
        most_fitted = sorted(selected_group, reverse=True, key=lambda u: u.evaluation)[0]
        
        if count <= sth:
            #if generation exceeds threshold, the sampling is stopped for saving the calculation cost.
            if count%100 == 0:
                print("sampling:{}".format(count))
                TIME_SERIES_NETWORK.append(most_fitted.getGenom())

        if abs(np.mean(selected_eva)) < 0.01: 
            print(most_fitted.getGenom())
            print("Quit GA:{}".format(count))
            return most_fitted.getGenom(), TIME_SERIES_NETWORK, DesiredGoal

    ## end of  main for statement
    print("End of run")
    return np.nan, np.nan, np.nan

print("Done")


"""
Execute Evolutionary Simulation
"""

print("TITLE{}".format(TITLE))
print("Rank{}".format(GoalMatrixRank))

import csv
import pandas as pd

global POPULATION_SIZE
global MAX_GENERATION
global nMatrix
global MUTATION_RATE

global nNode
global nLayer

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
ActiveNodeDefinition = 1

### GoalMatrixParameters
GoalMatrixRank = 1
GoalMatirxNorm = 60
GoalMatrixVariance = 10 # If this set to np.nan, the variance is not normalized.


### Goal matrix chnage
GoalMatrixChange = False

### Goal matrix expansion
GoalExpansion = False
DuplicateTime = 1000


END_NETWORK = list()
INITIAL_NETWORKS = list()
GOAL_MATRICES = list()

for g in [0.01, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80]:
    print("initial norm:{}".format(g))
    norm = g

    endnetwork = list()
    initialnetwork = list()
    goalmatrix = list()
    for i in range(100):
        Define_global_value_in_modules(nNode)
        DesiredGoal = CreateRandomGoalMatrix(GoalMatrixRank, norm = GoalMatirxNorm, zvar = GoalMatrixVariance)
        Define_global_value_in_modules(nNode, nLayer, GoalMatrixRank, POPULATION_SIZE, MUTATION_RATE, DesiredGoal, ActiveNodeDefinition)
        steadystate_network, initialstate_networks, goal_matrix = GeneticAlgorithm(norm, GoalMatrixRank, sth, DesiredGoal)
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

if sth > 0:
    INITIAL_NETWORKS = np.array(INITIAL_NETWORKS, dtype=object)
    np.save(f"{TITLE}_INIT", INITIAL_NETWORKS)


print("RunID:{} Successfully Done!!".format(TITLE))


