
from scipy.stats import special_ortho_group
from itertools import chain
import numpy as np
import random
import copy
import math
import sys
import os 

def Define_global_value_in_modules(nNode_=None, nLayer_=None, POPULATION_SIZE_=None, 
    MUTATION_RATE_=None, raw_mut_rate_ = None, ActiveNodeDefinition_ = None, EVALUATION_ = None, SELECTION_ = None):
    global nNode
    global nLayer
    global nMatrix
    global POPULATION_SIZE
    global MUTATION_RATE
    global raw_mut_rate
    global ActiveNodeDefinition
    global EVALUATION
    global SELECTION
    print("eva:{}".format(EVALUATION_))
    if nNode_ is not None: nNode = nNode_
    if nLayer_ is not None: nLayer, nMatrix = nLayer_, nLayer_-1
    if POPULATION_SIZE_ is not None: POPULATION_SIZE = POPULATION_SIZE_
    if MUTATION_RATE_ is not None: MUTATION_RATE = MUTATION_RATE_
    if raw_mut_rate_ is not None: raw_mut_rate=raw_mut_rate_ 
    ActiveNodeDefinition = 0 if ActiveNodeDefinition_ is None else ActiveNodeDefinition_
    EVALUATION = 0 if EVALUATION_ is None else EVALUATION_
    SELECTION = "tournament" if SELECTION_ is None else SELECTION_


def Define_global_goal(DesiredGoal_):
    global DesiredGoal
    DesiredGoal = DesiredGoal_



class genom:

    genom_list = None
    evaluation = None

    def __init__(self, genom_list, evaluation):
        self.genom_list = genom_list
        self.evaluation = evaluation


    def getGenom(self):
        return self.genom_list


    def getEvaluation(self):
        return self.evaluation


    def setGenom(self, genom_list):
        self.genom_list = genom_list


    def setEvaluation(self, evaluation):
        self.evaluation = evaluation

def create_network(nNode, nMatrix, norm):
    #This function returns Node x Node x Layer Matrix, which describe individual's network structure
    pre_net = np.random.uniform(0, 0.05, (nMatrix, nNode, nNode))
    pre_net_norm = (np.linalg.norm(Total_in_out(pre_net), ord="fro"))
    normalizeF = (norm/pre_net_norm)**(1/(nMatrix))
    network = normalizeF*pre_net
    return network

def Total_in_out(Ind):
    nMatrix = len(Ind)
    z = np.dot(Ind[nMatrix -1], Ind[nMatrix -2])
    for i in range(nMatrix -3, -1, -1):
        z = np.dot(z, Ind[i])
    return z

def add_mutation(Ind, corrected_mut_rate):   
    var = 0.1
    bitmask = np.where(np.random.uniform(0, 1, (nMatrix, nNode, nNode))  < corrected_mut_rate, 1, 0)
    noise_matrix = np.random.normal(1, var, (nMatrix, nNode, nNode))  
    pre_mult_mat = bitmask*noise_matrix
    mult_mat = np.where(pre_mult_mat == 0,  1, pre_mult_mat)
    mutated_network = Ind.getGenom()*mult_mat
    Ind.setGenom(mutated_network)
    return Ind


## Tournament selection
def select(population, tournament_size):    #Tournament selection

    if SELECTION == "tournament": #tournament selection
        tournament_groups = [random.sample(population, tournament_size) for i in range(POPULATION_SIZE)]
        selected = [ sorted(tournament_group, reverse=True, key=lambda u: u.evaluation)[0] for tournament_group in tournament_groups]
        return selected
    else: #elite selection
        upper_ind = sorted(population, reverse=True, key=lambda u: u.evaluation)[0:25]
        non_upper_ind = [ind for ind in population if ind not in upper_ind]
        random_ind = random.sample(non_upper_ind, 75)
        selected =upper_ind + random_ind
        return selected

def Deleted_fitness(Ind, node, layer):    
    modified_network =  copy.deepcopy(Ind)
    Input_link_layer = layer - 1
    Output_link_layer = layer
    if layer == 0: #Eliminate input layer
        modified_network[Output_link_layer][:,node] = 0  #eliminate output
    elif layer == nLayer-1: #Eliminate output layer
        modified_network[Input_link_layer][node,] = 0   #eliminate input
    else: #Eliminate intermidiate layer
        modified_network[Input_link_layer][node,] = 0   #eliminate input from 
        modified_network[Output_link_layer][:,node] = 0

    orig_network = copy.deepcopy(Ind)
    orig_fit = evaluation(orig_network, DesiredGoal)
    modi_fit = evaluation(modified_network, DesiredGoal)
    relative_fitness = abs(orig_fit-modi_fit)    
    return(relative_fitness)

def Deleted_totalinout(Ind, node, layer):
    modified_network =  copy.deepcopy(Ind)
    Input_link_layer = layer - 1
    Output_link_layer = layer
    if layer == 0: #Eliminate input layer
        modified_network[Output_link_layer][:,node] = 0  #eliminate output
    elif layer == nLayer-1: #Eliminate output layer
        modified_network[Input_link_layer][node,] = 0   #eliminate input
    else: #Eliminate intermidiate layer
        modified_network[Input_link_layer][node,] = 0   #eliminate input from 
        modified_network[Output_link_layer][:,node] = 0
    orig_network = copy.deepcopy(Ind)
    diff = Total_in_out(orig_network) - Total_in_out(modified_network)
    relative_fitness = np.sum(diff**2)  
    return(relative_fitness)


def MaximumInteraction(Ind, node, layer):
    inmax,outmax = 0,0
    if layer > 0:
        inmax = max(Ind[layer-1][node,:])
    if layer < nLayer-1:
        outmax = max(Ind[layer][:,node])
    #threshold: 0.05
    return max(inmax,outmax)

def Relative_fitness_in_layer(layer, mode, Ind):
    if ActiveNodeDefinition == 0:
        relative_fitness = [(Deleted_fitness(Ind, node, layer)) for node in range(nNode)]
        threshold = 0.001
    elif ActiveNodeDefinition == 1:
        relative_fitness = [(Deleted_totalinout(Ind, node, layer)) for node in range(nNode)]
        threshold = 0.001
    elif ActiveNodeDefinition == 2:
        relative_fitness = [(MaximumInteraction(Ind, node, layer)) for node in range(nNode)]
        threshold = 0.05
    relative_fitness_in_layer = relative_fitness/sum(relative_fitness)
    active_node_test = np.where(relative_fitness_in_layer > threshold,  1, 0)
    if mode == "result":
        return(sum(active_node_test))
    elif mode == "test":
        return(active_node_test)

def Active_node(Ind, mode):
    active_node_list = [Relative_fitness_in_layer(layer, mode, Ind) for layer in range(nLayer)]
    return(active_node_list)


def CreateRandomGoalMatrix(rank, norm, zvar = np.nan):
    
    if (nNode%2 != 0) & (rank != 1):
        raise ValueError("nNode must be even number when rank is larger than 1")

    if rank == 1:
        G = np.full((nNode,nNode),10)
    elif rank == 2:
        a = np.full((int(nNode/2),int(nNode/2)),10)
        b = np.full((int(nNode/2),int(nNode/2)), 0)
        G = np.block([[a,b],[b,a]])
    elif rank == 3:
        a = np.full((int(nNode/3),int(nNode/3)),10)
        b = np.full((int(nNode/3),int(nNode/3)), 0)
        G = np.block([[a,b,b],[b,a,b], [b, b,a]])
    elif rank == 4:
        a = np.full((2,2),10)
        b = np.full((2,2),0)
        c = np.diag([10,10])
        G = np.block([[a,b,b],[b,a,b],[b,b,c]])
    elif rank == 5:
        a = np.full((2,2),10)
        b = np.full((2,2),0)
        c = np.diag([10,10])
        G = np.block([[a,b,b],[b,c,b],[b,b,c]])   
    elif rank == nNode:
        G = np.diag(np.repeat(10,nNode))
    else:
        raise ValueError("Please choose rank from 1, 2, 3, and 6")
    
    # Search random matrix which only have positive values under the given condition
    for i in range(5000): 
        # randomize matrix
        rot_m = special_ortho_group.rvs(nNode)
        pre_DesiredGoal = np.dot(rot_m, G)
        pre_DesiredGoal[pre_DesiredGoal < 0] = -1*pre_DesiredGoal[pre_DesiredGoal < 0]
        #Uniform frobenius norm
        pre_goal_norm = np.linalg.norm(pre_DesiredGoal, ord="fro")
        DesiredGoal = (norm/pre_goal_norm)*pre_DesiredGoal
        #Normalize variance if zvar is given.
        if not np.isnan(zvar):
            # ref S1 text for this equasion
            z = (DesiredGoal - np.mean(DesiredGoal))*np.sqrt(zvar)/(np.std(DesiredGoal)) + np.sqrt(100-zvar)
            if len(z[z < 0]) <= 0:
                break
        else:
            z = DesiredGoal
            break
    if len(z[z < 0]) > 0:
        raise ValueError("Can not find matrix which all elements are positive")
    return(z)


def statistics_mode(data):
    uniqs, counts = np.unique(data, return_counts=True)
    mode = np.mean(uniqs[counts == np.amax(counts)]) 
    return(mode)

#Define fitness as the distance of goal matrix and total in-out matrix.

def evaluation(Ind_network, DesiredGoal):
    z = Total_in_out(Ind_network)
    fitness = -1*np.sum((z - DesiredGoal)**2) #+ np.sum(Ind_network**2) Penalty
    return fitness


def L1cost_evaluation(Ind_network, DesiredGoal):
    z = Total_in_out(Ind_network)
    fitness = -1*np.sum((z - DesiredGoal)**2) - np.sum(abs(Ind_network))
    return fitness


def L2cost_evaluation(Ind_network, DesiredGoal):
    z = Total_in_out(Ind_network)
    fitness = -1*np.sum((z - DesiredGoal)**2) - np.sum(Ind_network**2) 
    return fitness


def set_eva(Ind, DesiredGoal):
    if EVALUATION == 0:
        eva = evaluation(Ind.getGenom(), DesiredGoal)
    elif EVALUATION == 1:
        eva = L1cost_evaluation(Ind.getGenom(), DesiredGoal)
    elif EVALUATION == 2:
        eva = L2cost_evaluation(Ind.getGenom(), DesiredGoal)
    Ind.setEvaluation(eva)
    return(Ind)



print("functios are loaded")


## For expansion of inputs and outputs
def CreateVirtualDuplicateInd(Ind, DesiredGoal):
    zeros = np.full((nNode,nNode),0.001)
    DuplicateNet = list()
    for LayerInt in Ind.getGenom():
        DuplicateNet.append(np.block([[LayerInt,zeros],[zeros,zeros]]))
    DuplicateNet = np.array(DuplicateNet)
    duplicate_genom = genom(DuplicateNet, evaluation(DuplicateNet, DesiredGoal))
    return duplicate_genom

def Duplicate(network, nNode): 
    DuplicateNet = list()
    for LayerInt in network:
        DuplicateNet.append(np.block([[LayerInt,LayerInt],[LayerInt,LayerInt]]))
    DuplicateNet = np.array(DuplicateNet)
    return DuplicateNet


def GeneticAlgorithm(norm, G_params, MAX_GENERATION, output_style, sth = -1, FLUCTUATION_MODE=-1, DuplicateTime=-1):

    global nNode
    global nLayer
    global nMatrix
    global POPULATION_SIZE
    global MUTATION_RATE
    global raw_mut_rate
    global ActiveNodeDefinition
    global EVALUATION
    global SELECTION

    print("nNode{}".format(nNode))
    DesiredGoal = CreateRandomGoalMatrix(rank=G_params[0], norm=G_params[1], zvar=G_params[2])
    Define_global_goal(DesiredGoal_ = DesiredGoal)

    print("goal norm: {}".format(np.linalg.norm(DesiredGoal, "fro")))
    print("goal variance: {}".format(np.var(DesiredGoal)))
    print("goal rank: {}".format(np.linalg.matrix_rank(DesiredGoal)))

    current_generation_individual_group_pre = [genom(create_network(nNode,nMatrix,norm),0) for i in range(POPULATION_SIZE)]
    current_generation_individual_group = [set_eva(Ind, DesiredGoal) for Ind in current_generation_individual_group_pre]
    
    initial_network_size = np.mean([np.linalg.norm(Total_in_out(Ind.getGenom()), 'fro') for Ind in current_generation_individual_group])
    
    print("initial network norm: {}".format(initial_network_size))

    PreGenFitness = 0
    
    #TIME_SERIES_NETWORK = list()

    if output_style == 1:
        active_node_list = list()
    if output_style == 2:
        TIME_SERIES_NETWORK = list()

    print("GA starts")

    for count in range(MAX_GENERATION):
        
        if count == DuplicateTime:
            #Duplicate
            print("Duplicate")
            zeros = np.full((nNode,nNode),0.001)
            DesiredGoal = Duplicate([DesiredGoal], nNode)[0]
            current_generation_individual_group = [
                CreateVirtualDuplicateInd(current_generation_individual_group[i], DesiredGoal) for i in range(POPULATION_SIZE)]
            
            nNode = nNode*2
            mutation_rate = raw_mut_rate/(nMatrix*nNode*nNode)
            Define_global_goal(DesiredGoal_ = DesiredGoal)
            Define_global_value_in_modules(nNode, nLayer, POPULATION_SIZE, MUTATION_RATE, raw_mut_rate, ActiveNodeDefinition, EVALUATION)

            print("Goal matrix:{}".format(DesiredGoal.shape))
            print("Network:{}\n".format(Total_in_out(current_generation_individual_group[0].getGenom()).shape))

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
        
        if count%100==0:
            most_fitted = sorted(selected_group, reverse=True, key=lambda u: u.evaluation)[0]
            active_node = Active_node(most_fitted.getGenom(), mode="result")
            ave_fitness = np.mean(selected_eva)
            print("Gen{}, Loss:{}, Active node: {}".format(count, ave_fitness, active_node))

            if output_style == 1:
                output_arr = active_node.copy()
                output_arr.append(ave_fitness)
                active_node_list.append(output_arr)

            if (output_style ==2) & (count <= sth):
            #if generation exceeds threshold, the sampling is stopped for saving the calculation cost.
                print("sampling:{}".format(count))
                TIME_SERIES_NETWORK.append(most_fitted.getGenom())

        if count%1000==0:
            ave_fitness = np.mean(selected_eva)
            if (FLUCTUATION_MODE == 0) & (abs(ave_fitness) < 0.01):
                print("Goal matrix change")
                DesiredGoal =  CreateRandomGoalMatrix(rank=G_params[0], norm=G_params[1], zvar=G_params[2])
                current_generation_individual_group = [set_eva(Ind, DesiredGoal) for Ind in current_generation_individual_group]
                Define_global_goal(DesiredGoal_ = DesiredGoal)

            if (FLUCTUATION_MODE == 1) & (abs(ave_fitness) < 0.01):
                print("Goal matrix change")
                DesiredGoal =  CreateRandomGoalMatrix(rank=6, norm=G_params[1], zvar=G_params[2])
                current_generation_individual_group = [set_eva(Ind, DesiredGoal) for Ind in current_generation_individual_group]
                Define_global_goal(DesiredGoal_ = DesiredGoal)

            if FLUCTUATION_MODE == 2:
                print("Goal matrix change")
                DesiredGoal =  CreateRandomGoalMatrix(rank=G_params[0], norm=G_params[1], zvar=G_params[2])
                current_generation_individual_group = [set_eva(Ind, DesiredGoal) for Ind in current_generation_individual_group]
                Define_global_goal(DesiredGoal_ = DesiredGoal)

        # Steady state
        if (abs(np.mean(selected_eva)) < 0.01) & (output_style%2 == 0): # no need to waist until steady state in output_style 1 and 3
            most_fitted = sorted(selected_group, reverse=True, key=lambda u: u.evaluation)[0]
            active_node = Active_node(most_fitted.getGenom(), mode="result")
            print("Saturation at {}".format(count))
            print("Gen{}, Fitness:{}, Active node: {}\n\n".format(count, np.mean(selected_eva), active_node))

            if output_style == 0:
                active_node = Active_node(most_fitted.getGenom(), mode="result")
                return active_node

            if output_style == 2:
                return most_fitted.getGenom(), TIME_SERIES_NETWORK, DesiredGoal

    ## end of  main for statement
    print("End of run")

    if output_style == 0:
        print("Not saturated. Results are ignored.")
        return [np.nan]*nLayer
    if output_style == 1:
        print("Simulation end")
        return active_node_list
    if output_style == 2:
        print("Not saturated. Results are ignored.")
        return [np.nan, np.nan, np.nan]
    if output_style == 3:
        print("Simulation end")
        return active_node


## Gradient descend 
def dLdW(network, AminusG, L):
    network_copy = copy.deepcopy(network)
    mat_dot = np.eye(nNode)
    for l in range(L+1, nMatrix):
        mat_dot  = np.dot(mat_dot, np.transpose(network_copy[l]))
    mat_dot = np.dot(mat_dot, AminusG)
    min_mat = 0#np.min(np.delete(np.array(range(nMatrix)), L))
    for l in range(0, L):
        mat_dot  = np.dot(mat_dot, np.transpose(network_copy[l]))
    return(mat_dot)

def F(network):
    AminusG = Total_in_out(copy.deepcopy(network))-DesiredGoal
    grad = 2*np.array([dLdW(network, AminusG, l) for l in range(nMatrix)])
    return grad*network


def GradientDescent(norm, G_params, MAX_STEP, output_style):

    DesiredGoal = CreateRandomGoalMatrix(rank=G_params[0], norm=G_params[1], zvar=G_params[2])
    Define_global_goal(DesiredGoal_ = DesiredGoal)

    print("goal norm: {}".format(np.linalg.norm(DesiredGoal, "fro")))
    print("goal variance: {}".format(np.var(DesiredGoal)))
    print("goal rank: {}".format(np.linalg.matrix_rank(DesiredGoal)))

    network = create_network(nNode, nMatrix, norm)
    initial_network_size = np.linalg.norm(Total_in_out(network), ord=2)
    print("initial network norm: {}".format(initial_network_size))
    
    h = 0.0001 # step width of Runge kutta

    if output_style == 1:
        active_node_list = list()

    for step in range(MAX_STEP):
        
        #### 4K Runge-Kutta ####
        k_1 = h * F(network)
        k_2 = h * F(network -  k_1 /2)
        k_3 = h * F(network -  k_2 /2)
        k_4 = h * F(network - k_3)
        network = network - 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4 )
        #######################
        loss = evaluation(network , DesiredGoal)
        
        if np.isnan(loss):
            print("Divergence")
            break

        if step%5000==0:
            print("========={}=========".format(step))
            active_node = Active_node(network, "result")
            print("Step{}, Loss:{}, Active node: {}\n\n".format(step, loss, active_node))

        if (step&100 == 0) & (output_style == 1):
            output_arr = active_node.copy()
            output_arr.append(loss)
            active_node_list.append(output_arr)


        if (abs(loss) < 0.01) & (output_style == 0): #Fitness < 0.01
            print("========={}=========".format(step))
            print("Saturation at {}".format(step))
            active_node = Active_node(network, "result")
            print("Step{}, Loss:{}, Active node: {}\n\n".format(step, loss, active_node))

            active_node = Active_node(network, mode="result")
            return active_node

    print("End of run")

    if output_style == 0:
        print("Not saturated. Results are ignored.")
        return [np.nan]*nLayer
    if output_style == 1:
        print("Simulation end")
        return active_node_list
