import numpy as np
import csv
import pandas as pd
from BasicFunctions import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="Output file name", type=str)
parser.add_argument("goal_matrix_rank", help="Goal matrix rank", type=int)
parser.add_argument("--goal_matrix_variance", help="Normalized goal matrix variance (No normalization if this set empty))", type=float)
parser.add_argument("--init_net_norm", help="Norm of total inout relation matrix at first step", type=float)
parser.add_argument("--algorithm", help="(GA, GD) Choose GA(GeneticAlgorithm; default) or GD(GradientDescent).", type=str)
parser.add_argument("--evaluation", help="(0,1,2) Set Evaluate function. 0: normal evaluation function (default), 1: L1 regularization, 2: L2 regularization", type=int)
parser.add_argument("--active_node", help="(0,1,2) Set Active node definition. 0: Contribution to fitness (default), 1: Contribution to total in-out, 2: Maximum interaction", type=int)
parser.add_argument("--number_of_node", help="Please specify the number of node (only even number)", type=int)
parser.add_argument("--number_of_layer", help="number of layer", type=int)
parser.add_argument("--max_generation", help="max count that simulation can reaches", type=int)
parser.add_argument("--sampling_period_in_early_phase", help="sampling period in early phase of evolution", type=int)

args = parser.parse_args()

TITLE = args.filename
if args.init_net_norm is not None:
    print("!!Warning: init_net_norm is ignored in this program")
INIT_NET_NORM = args.init_net_norm

print("filename:{}".format(TITLE))

#### Simulation parameters #####

if args.algorithm == "GD":
    print("!!Warning: Only GA is allowed in this program")

### Simulaton Algorithm
### GA: Genetic Algorithm (Evolution mode)
### GD: Gradient Descent (ODE model)

EVALUATION = 0 if args.evaluation is None else args.evaluation
### Evaluation function
### 0. Distance between total in-out matrix and goal matrix
### 1. Distance between total in-out matrix and goal matrix + L1 regularization
### 2. Distance between total in-out matrix and goal matrix + L2 regularization

ActiveNodeDefinition = 0 if args.active_node is None else args.active_node
### Active node definition
### 0. Relative contribution of each node to the fitness.
### 1. Relative contribution of each node to the total in-out relation.
### 2. Relative strength of maximal interaction of each node.

GoalMatrixRank = int(args.goal_matrix_rank)
GoalMatrixNorm = 60
GoalMatrixVariance = np.nan if args.goal_matrix_variance is None else args.goal_matrix_variance
G_params = [GoalMatrixRank, GoalMatrixNorm, GoalMatrixVariance]
### GoalMatrixParameters

nNode = 6 if args.number_of_node is None else args.number_of_node
nLayer = 5 if args.number_of_layer is None else args.number_of_layer
nMatrix = nLayer-1
### Network structure


POPULATION_SIZE = 100
MAX_GENERATION = 50000 if args.max_generation is None else args.max_generation
MUTATION_RATE = 0.2/(nMatrix*nNode*nNode)

if args.sampling_period_in_early_phase == None:
    raise ValueError("Please specify the sampling_period_in_early_phase (e.g., --sampling_period_in_early_phase 10000")

sth = args.sampling_period_in_early_phase

Define_global_value_in_modules(nNode, nLayer, POPULATION_SIZE, MUTATION_RATE, ActiveNodeDefinition)

END_NETWORK = list()
INITIAL_NETWORKS = list()
GOAL_MATRICES = list()


init_value = [0.01, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80]
for init_net_norm in init_value:
    endnetwork = list()
    initialnetwork = list()
    goalmatrix = list()
    for _ in range(100):
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


