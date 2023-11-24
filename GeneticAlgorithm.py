def GeneticAlgorithm(norm, GoalMatrixRank, sth, DesiredGoal, output_style):

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
            if output_style == 0:
                ActiveNode = Active_node(most_fitted, mode="result")
                return ActiveNode

            return most_fitted.getGenom(), TIME_SERIES_NETWORK, DesiredGoal

    ## end of  main for statement
    print("End of run")
    return np.nan, np.nan, np.nan

print("GeneticAlgorithm is loaded from GeneticAlgorithm.py")

