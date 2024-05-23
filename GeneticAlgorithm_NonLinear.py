print("Loaded")

import sys
import numpy as np
import copy
import matplotlib.pyplot as plt
import random
import math
np.set_printoptions(precision=4, floatmode='maxprec')
np.set_printoptions(suppress=True)
import os 

from sklearn.datasets import load_digits

print("Loaded")

nNode = 6
nMatrix = 3
nLayer = 4

class genom:

    genom_list = None
    evaluation = None
    input_layer = None

    def __init__(self, input_layer, genom_list, evaluation):
        self.genom_list = genom_list
        self.evaluation = evaluation
        self.input_layer = input_layer


    def getGenom(self):
        return self.genom_list
    
    def getInputLayer(self):
        return self.input_layer

    def getEvaluation(self):
        return self.evaluation

    def setGenom(self, genom_list):
        self.genom_list = genom_list
    
    def setInputLayer(self, input_layer):
        self.input_layer = input_layer
        
    def setEvaluation(self, evaluation):
        self.evaluation = evaluation

def Total_in_out(Ind):
    z = np.dot(Ind[nMatrix -1], Ind[nMatrix -2])
    for i in range(nMatrix -3, -1, -1):
        z = np.dot(z, Ind[i])
    return z


def CreateInd(nNode, nMatrix, norm):
    #This function returns Node x Node x Layer Matrix, which describe individual's network structure
    pre_input_layer = np.random.uniform(-1, 1, (nNode, 64))
    pre_net =  np.random.uniform(-1, 1, (nMatrix, nNode, nNode))
    out = np.dot(Total_in_out(pre_net), pre_input_layer)
    pre_net_norm = (np.linalg.norm(out, ord="fro")) 
    normalizeF = (norm/pre_net_norm)**(1/(nMatrix+1))
    network = normalizeF*pre_net
    input_layer = normalizeF*pre_input_layer
    return genom(input_layer, network, 0)


def add_mutation(Ind, corrected_mut_rate):

    var = 0.5
    bitmask = np.where(np.random.uniform(0, 1, (nMatrix, nNode, nNode))  < corrected_mut_rate, 1, 0)
    noise_matrix = np.random.normal(1, var, (nMatrix, nNode, nNode))
    pre_mult_mat = bitmask*noise_matrix
    mult_mat = np.where(pre_mult_mat == 0,  1, pre_mult_mat)
    mutated_network = Ind.getGenom()*mult_mat
    Ind.setGenom(mutated_network)
    
    bitmask = np.where(np.random.uniform(0, 1, (nNode, 64))  < corrected_mut_rate, 1, 0)
    noise_matrix = np.random.normal(1, var, (nNode, 64))
    pre_mult_mat = bitmask*noise_matrix
    mult_mat = np.where(pre_mult_mat == 0,  1, pre_mult_mat)
    mutated_input_layer = Ind.getInputLayer()*mult_mat
    Ind.setInputLayer(mutated_input_layer)

    return Ind

def select(population, tournament_size):    
    #Tournament selection
    tournament_groups = [random.sample(population, tournament_size) for i in range(POPULATION_SIZE)]
    selected = [ sorted(tournament_group, reverse=True, key=lambda u: u.evaluation)[0] for tournament_group in tournament_groups]
    return selected

def Deleted_fitness(Ind, node, layer):
    modified_network =  copy.deepcopy(Ind.getGenom())
    modified_input_layer = copy.deepcopy(Ind.getInputLayer())
    
    Input_link_layer = layer - 1
    Output_link_layer = layer
    
    if layer == -1: #Eliminate input layer
        modified_input_layer[:,node] = 0  #eliminate output
    elif layer == 0:
        modified_input_layer[node,] = 0  #eliminate output
        modified_network[0][:,node] = 0
    elif layer == nLayer-1: #Eliminate output layer
        modified_network[Input_link_layer][node,] = 0   #eliminate input
    else: #Eliminate intermidiate layer
        modified_network[Input_link_layer][node,] = 0   #eliminate input from 
        modified_network[Output_link_layer][:,node] = 0
    orig_fit = Ind.getEvaluation()
    modi_fit = evaluation(modified_input_layer, modified_network)
    relative_fitness = abs(orig_fit-modi_fit)

    return(relative_fitness)

def Relative_fitness_in_layer(layer, mode, Ind):
    if layer == -1:
        relative_fitness = [(Deleted_fitness(Ind, node, layer)) for node in range(64)]
        if np.sum(relative_fitness) != 0:
            relative_fitness_in_layer = np.array(relative_fitness)/np.sum(relative_fitness)
        else:
            relative_fitness_in_layer = np.repeat(1,nNode)
        active_node_test = np.where(relative_fitness_in_layer > 0.001/6,  1, 0)
        #node数による補正をかける
    else:
        relative_fitness = [(Deleted_fitness(Ind, node, layer)) for node in range(nNode)]
        if np.sum(relative_fitness) != 0:
            relative_fitness_in_layer = np.array(relative_fitness)/np.sum(relative_fitness)
        else:
            relative_fitness_in_layer = np.repeat(1,nNode)
        active_node_test = np.where(relative_fitness_in_layer > 0.001,  1, 0)

    if mode == "result":
        return(sum(active_node_test))
    elif mode == "test":
        return(active_node_test)

def Active_node_in_MATLAB(Ind, mode):
    #orig_fit = abs(evaluation(Ind.getGenom(), DesiredGoal))
    orig_fit = abs(Ind.getEvaluation())
    if orig_fit == 0:
        orig_fit = 0.001
    active_node_list = [Relative_fitness_in_layer(layer, mode, Ind) for layer in range(-1,nLayer)]
    return(active_node_list)

digits = load_digits()
DIG_target = list(digits.target)[0:300]
tar_2 = [i for i, x in enumerate(DIG_target) if x == 2]
tar_4 = [i for i, x in enumerate(DIG_target) if x == 4]
tar_6 = [i for i, x in enumerate(DIG_target) if x == 6]
tar_8 = [i for i, x in enumerate(DIG_target) if x == 8]
tar_index = tar_2 + tar_4 + tar_6 + tar_8
tar_sindex = sorted(tar_index)
DIG_INPUT = digits.data[tar_sindex,]
DIG_TRUE = digits.target[tar_sindex]
DATA_SIZE = len(DIG_TRUE)

def set_func(x):
    set_result = np.zeros(6)
    if x in [2,4]: set_result[0] = 1
    if x in [2,6]: set_result[1] = 1
    if x in [2,8]: set_result[2] = 1
    if x in [4,6]: set_result[3] = 1
    if x in [4,8]: set_result[4] = 1
    if x in [6,8]: set_result[5] = 1
    return set_result

DIG_prob_arr = np.ones((nNode,DATA_SIZE))
for i in range(DATA_SIZE):
    DIG_prob_arr[:, i] = set_func(DIG_TRUE[i])

def tanh(x):
    y = (np.exp(x)- np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return y

def F(x):
    x_ = copy.deepcopy(x)
    x_ = x_
    x_[np.where(x_ > 700)] = 700
    x_[np.where(x_ < -700)] = -700
    #for prohibitting overflow.
    return (1 + tanh(x_))/2

def Output(input_layer, middle_network, SeeRawVal=False):
    if SeeRawVal:
        V = (DIG_INPUT.T)[:,0]
    else:
        V = DIG_INPUT.T
    U = F(np.dot(input_layer, V))
    for i in range(len(middle_network)):
        if SeeRawVal & (i == len(middle_network)-1):
            return np.dot(middle_network[i], U)
        U = F(np.dot(middle_network[i], U))
    return U

# Non linear evaluate function
def evaluation(input_layer, middle_network):
    out = Output(input_layer, middle_network)
    eva = -1*np.sum((DIG_prob_arr - out)**2)
    return(eva)

def set_eva(Ind):
    network = Ind.getGenom()
    eva = evaluation(Ind.getInputLayer(), Ind.getGenom()) #Normal
    Ind.setEvaluation(eva)
    return(Ind)


POPULATION_SIZE = 1000
MAX_GENERATION = 500000

MUTATION_RATE = 0.2/((nMatrix*nNode*nNode) + 64*nNode)
Initialization = True
NETWORK_NORM = float(sys.argv[1])
print(NETWORK_NORM)

current_generation_individual_group_pre = [CreateInd(nNode,nMatrix,NETWORK_NORM) for i in range(POPULATION_SIZE)]
current_generation_individual_group = [set_eva(Ind) for Ind in  current_generation_individual_group_pre]
meanint = np.mean(current_generation_individual_group[0].getGenom()[0])
ActiveNodeList = list()
end = 20000
for count in range(end):  #Repeat generation 

    next_generation_network = [copy.deepcopy(Ind.getGenom()) for Ind in current_generation_individual_group]
    next_geenration_input =  [copy.deepcopy(Ind.getInputLayer()) for Ind in current_generation_individual_group]
    #next_geenration_Thr =  [copy.deepcopy(Ind.getThr()) for Ind in current_generation_individual_group]
    next_generation_individual_group = [genom(input_layer, network, 0) for input_layer, network in zip(next_geenration_input, next_generation_network)]
    
    next_generation_individual_group_mutated = [add_mutation(Ind, MUTATION_RATE) for Ind in next_generation_individual_group]
    next_generation_individual_group_evaluated = [set_eva(Ind) for Ind in  next_generation_individual_group_mutated]

    #Selection
    mixed_population = current_generation_individual_group + next_generation_individual_group_evaluated
    #tournament size is 4
    selected_group = select(mixed_population, 4)

    #Generation change
    selected_inp = [copy.deepcopy(Ind.getInputLayer()) for Ind in selected_group]
    selected_net = [copy.deepcopy(Ind.getGenom()) for Ind in selected_group]
    selected_eva = [Ind.getEvaluation() for Ind in selected_group]
    #selected_Thr = [Ind.getThr() for Ind in selected_group]
    current_generation_individual_group = [genom(selected_inp[i], selected_net[i], selected_eva[i]) for i in range(POPULATION_SIZE)]
    
    #most_fitted = sorted(selected_group, reverse=True, key=lambda u: u.evaluation)[0]
    #ConsActiveNode = Active_node_in_MATLAB(most_fitted, mode="result")

        
    if count%100==0:
        if np.isnan(np.mean(selected_eva)):
            print("nan detected. break") 
            break
        #LossLixt.append(np.mean(selected_eva))
        #Output section for debugging
        most_fitted = sorted(selected_group, reverse=True, key=lambda u: u.evaluation)[0]
        #print("average interaction:{}".format(np.mean(most_fitted.getGenom())))
        ConsActiveNode = Active_node_in_MATLAB(most_fitted, mode="result")
        ConsActiveNode.append(np.mean(selected_eva))
        mfout = Output(most_fitted.getInputLayer(), most_fitted.getGenom())
        score_matrix = np.round(mfout) - DIG_prob_arr
        accur_score = 1.0-np.sum(abs(score_matrix))/(score_matrix.shape[0]*score_matrix.shape[1])
        ConsActiveNode.append(accur_score)
        ActiveNodeList.append(ConsActiveNode)        
        #print("\nGen{}, Ave Loss:{}, MF Loss:{},  \nOutVal:{}, \nActive node:\n{}".format(count, np.mean(selected_eva), most_fitted.getEvaluation() , mfout, ConsActiveNode))

TITLE = sys.argv[2]
np.save(f"S13/NonLinearA0_{TITLE}", ActiveNodeList)
print("End")
