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
parser.add_argument("--raw_mut_rate", help="mutatino rate, default: 0.2", type=float)
args = parser.parse_args()



TITLE = args.filename
if args.init_net_norm is not None:
    print("init_net_norm is ignored in this program")
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

GoalMatrixRank = int(args.goal_matrix_rank)
GoalMatrixNorm = 60
GoalMatrixVariance = np.nan if args.goal_matrix_variance is None else args.goal_matrix_variance
G_params = [GoalMatrixRank, GoalMatrixNorm, GoalMatrixVariance]
### GoalMatrixParameters

nNode = 6 if args.number_of_node is None else args.number_of_node
nLayer = 5 if args.number_of_layer is None else args.number_of_layer
nMatrix = nLayer-1
### Network structure

raw_mut_rate = 0.2 if args.raw_mut_rate is None else args.raw_mut_rate

print("ALGORITHM: {}".format(ALGORITHM))
print("Rank:{}".format(GoalMatrixRank))


if ALGORITHM == "GA":
    POPULATION_SIZE = 100
    MAX_GENERATION = 50000 if args.max_generation is None else args.max_generation
    MUTATION_RATE = raw_mut_rate/(nMatrix*nNode*nNode)
    init_value =  [0.01,0.1,1.0,2.0,4.0,6.0,8.0,10,20,30,40,50,60,70,80] #for fig3B
    #initvalue = [0.01, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80] 
    Define_global_value_in_modules(nNode, nLayer, POPULATION_SIZE, MUTATION_RATE, raw_mut_rate, ActiveNodeDefinition, EVALUATION_=EVALUATION)
if ALGORITHM == "GD":
    MAX_STEP = 10000000 if args.max_generation is None else args.max_generation
    init_value =  [0.001, 0.01,0.1,1.0,2.0,4.0,6.0,8.0,10,20,30,40,50,60,70,80]
    Define_global_value_in_modules(nNode_ = nNode, nLayer_ = nLayer, ActiveNodeDefinition_ = ActiveNodeDefinition, EVALUATION_=EVALUATION)


waist_std_list = list()
waist_mean_list = list()
waist_mode_list = list()
for init_net_norm in init_value:
    if ALGORITHM == "GA":
        waist_size_set = np.array(
            [np.min(GeneticAlgorithm(init_net_norm, G_params, MAX_GENERATION, output_style = 0)) for _ in range(100)]
            )
    if ALGORITHM == "GD":
        waist_size_set = np.array(
            [np.min(GradientDescent(init_net_norm, G_params, MAX_STEP, output_style = 0)) for _ in range(100)]
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
