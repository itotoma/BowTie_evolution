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
parser.add_argument("--max_generation", help="max count that simulation can reaches", type=int)
parser.add_argument("--selection_method", help="Choose selection method in GA (tournament, elite) (tournament if this set empty))", type=str) 
parser.add_argument("--skip_saturation_stop", help="GA runs until max generation if you set 3", type=str) 

args = parser.parse_args()



TITLE = args.filename
if args.init_net_norm is None:
    raise ValueError("Please specify the init net norm (e.g., --init_net_norm 0.01)")
INIT_NET_NORM = args.init_net_norm

print("filename:{}".format(TITLE))

#### Simulation parameters #####

ALGORITHM = "GA" if args.algorithm is None else args.algorithm
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

selection_method = "tournament" if args.selection_method is None else args.selection_method
### Selection method in GA

output_style = 0 if args.skip_saturation_stop is None else 3


GoalMatrixRank = int(args.goal_matrix_rank)
GoalMatrixNorm = 60
GoalMatrixVariance = np.nan if args.goal_matrix_variance is None else args.goal_matrix_variance
G_params = [GoalMatrixRank, GoalMatrixNorm, GoalMatrixVariance]
### GoalMatrixParameters

nNode = 6 if args.number_of_node is None else args.number_of_node
nLayer=5
nMatrix = nLayer-1
### Network structure

print("ALGORITHM: {}".format(ALGORITHM))
print("Rank:{}".format(GoalMatrixRank))


if ALGORITHM == "GA":
    POPULATION_SIZE = 100
    MAX_GENERATION = 50000 if args.max_generation is None else args.max_generation
    MUTATION_RATE = 0.2/(nMatrix*nNode*nNode)
    Define_global_value_in_modules(nNode, nLayer, POPULATION_SIZE, MUTATION_RATE, ActiveNodeDefinition, EVALUATION_=EVALUATION)
if ALGORITHM == "GD":
    MAX_STEP = 10000000 if args.max_generation is None else args.max_generation
    Define_global_value_in_modules(nNode_ = nNode, nLayer_ = nLayer, ActiveNodeDefinition_ = ActiveNodeDefinition, EVALUATION_=EVALUATION)

active_node_set = list()

for _ in range(100):

    if ALGORITHM == "GA":
        active_node = GeneticAlgorithm(INIT_NET_NORM, G_params, MAX_GENERATION, output_style = output_style)
    if ALGORITHM == "GD":
        active_node = GradientDescent(INIT_NET_NORM, G_params, MAX_STEP, output_style = output_style)
    active_node_set.append(active_node)

active_node_data = np.array(active_node_set)
filename = TITLE + ".txt"
np.savetxt(filename, active_node_data)


