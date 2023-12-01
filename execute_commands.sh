#!/bin/sh

# This shell file enumerates commands which generates data analyzed in the Itoh et al., 2023.
# Since each command takes time to complete, execution in high performance computer is preffered.

# Note
# Some scripts lost unneccesary information for generating corresponding figure. 
# If you want to obtain full information (e.g., evolutionary trajectory of network structure and fitness), please run DynamicsInitDependncy.py (GeneticAlgorithm function returns full information by setting output_style=2).

#### Fig3A Analyzed data ####
mkdir Fig3A
python3 RankDependency.py "Fig3A/RankDepend_R1_A001" 1 --init_net_norm 0.01 
python3 RankDependency.py "Fig3A/RankDepend_R2_A001" 2 --init_net_norm 0.01
python3 RankDependency.py "Fig3A/RankDepend_R3_A001" 3 --init_net_norm 0.01
python3 RankDependency.py "Fig3A/RankDepend_R6_A001" 6 --init_net_norm 0.01


#### Fig3B Analyzed data ####
mkdir Fig3B
python3 InitDependency.py "Fig3B/InitDepend_R1" 1
python3 InitDependency.py "Fig3B/InitDepend_R2" 2
python3 InitDependency.py "Fig3B/InitDepend_R3" 3
python3 InitDependency.py "Fig3B/InitDepend_R6" 6

#### S4FigA Analyzed data ####
mkdir S4FigA
python3 InitDependency.py "S4FigA/InitDepend_R1_VN" 1 --goal_matrix_variance 1
python3 InitDependency.py "S4FigA/InitDepend_R2_VN" 2 --goal_matrix_variance 1
python3 InitDependency.py "S4FigA/InitDepend_R3_VN" 3 --goal_matrix_variance 1
python3 InitDependency.py "S4FigA/InitDepend_R6_VN" 6 --goal_matrix_variance 1


#### Fig3CD Analyzed data ####
mkdir Fig3CD
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA001_R1" 1 --init_net_norm 0.01 --max_generation 10000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA001_R2" 2 --init_net_norm 0.01 --max_generation 15000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA001_R3" 3 --init_net_norm 0.01 --max_generation 80000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA001_R6" 6 --init_net_norm 0.01 --max_generation 150000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA40_R1" 1 --init_net_norm 40 --max_generation 10000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA40_R2" 2 --init_net_norm 40 --max_generation 15000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA40_R3" 3 --init_net_norm 40 --max_generation 80000 
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA40_R6" 6 --init_net_norm 40 --max_generation 150000

#### S3FigAB Analyzed data ####
mkdir S3FigAB
python3 EvolTrajectory.py "S3FigAB/EvolTrajectoryA001_R1_AN1" 1 --init_net_norm 0.01 --max_generation 10000 --active_node 1
python3 EvolTrajectory.py "S3FigAB/EvolTrajectoryA001_R2_AN1" 2 --init_net_norm 0.01 --max_generation 15000 --active_node 1
python3 EvolTrajectory.py "S3FigAB/EvolTrajectoryA001_R3_AN1" 3 --init_net_norm 0.01 --max_generation 80000 --active_node 1
python3 EvolTrajectory.py "S3FigAB/EvolTrajectoryA001_R6_AN1" 6 --init_net_norm 0.01 --max_generation 150000 --active_node 1
python3 EvolTrajectory.py "S3FigAB/EvolTrajectoryA40_R1_AN1" 1 --init_net_norm 40 --max_generation 10000 --active_node 1
python3 EvolTrajectory.py "S3FigAB/EvolTrajectoryA40_R2_AN1" 2 --init_net_norm 40 --max_generation 15000 --active_node 1
python3 EvolTrajectory.py "S3FigAB/EvolTrajectoryA40_R3_AN1" 3 --init_net_norm 40 --max_generation 80000 --active_node 1
python3 EvolTrajectory.py "S3FigAB/EvolTrajectoryA40_R6_AN1" 6 --init_net_norm 40 --max_generation 150000 --active_node 1


#### S3FigCD Analyzed data ####
mkdir S3FigCD
python3 EvolTrajectory.py "S3FigCD/EvolTrajectoryA001_R1_AN2" 1 --init_net_norm 0.01 --max_generation 10000 --active_node 2
python3 EvolTrajectory.py "S3FigCD/EvolTrajectoryA001_R2_AN2" 2 --init_net_norm 0.01 --max_generation 15000 --active_node 2
python3 EvolTrajectory.py "S3FigCD/EvolTrajectoryA001_R3_AN2" 3 --init_net_norm 0.01 --max_generation 80000 --active_node 2
python3 EvolTrajectory.py "S3FigCD/EvolTrajectoryA001_R6_AN2" 6 --init_net_norm 0.01 --max_generation 150000 --active_node 2
python3 EvolTrajectory.py "S3FigCD/EvolTrajectoryA40_R1_AN2" 1 --init_net_norm 40 --max_generation 10000 --active_node 2
python3 EvolTrajectory.py "S3FigCD/EvolTrajectoryA40_R2_AN2" 2 --init_net_norm 40 --max_generation 15000 --active_node 2
python3 EvolTrajectory.py "S3FigCD/EvolTrajectoryA40_R3_AN2" 3 --init_net_norm 40 --max_generation 80000 --active_node 2
python3 EvolTrajectory.py "S3FigCD/EvolTrajectoryA40_R6_AN2" 6 --init_net_norm 40 --max_generation 150000 --active_node 2


#### S4FigCD Analyzed data ####
mkdir S4FigCD
python3 EvolTrajectory.py "S4FigCD/EvolTrajectoryA001_R1_VN" 1 --init_net_norm 0.01 --max_generation 10000 --goal_matrix_variance 1
python3 EvolTrajectory.py "S4FigCD/EvolTrajectoryA001_R2_VN" 2 --init_net_norm 0.01 --max_generation 15000 --goal_matrix_variance 1
python3 EvolTrajectory.py "S4FigCD/EvolTrajectoryA001_R3_VN" 3 --init_net_norm 0.01 --max_generation 80000 --goal_matrix_variance 1
python3 EvolTrajectory.py "S4FigCD/EvolTrajectoryA001_R6_VN" 6 --init_net_norm 0.01 --max_generation 150000 --goal_matrix_variance 1
python3 EvolTrajectory.py "S4FigCD/EvolTrajectoryA40_R1_VN" 1 --init_net_norm 40 --max_generation 10000 --goal_matrix_variance 1
python3 EvolTrajectory.py "S4FigCD/EvolTrajectoryA40_R2_VN" 2 --init_net_norm 40 --max_generation 15000 --goal_matrix_variance 1
python3 EvolTrajectory.py "S4FigCD/EvolTrajectoryA40_R3_VN" 3 --init_net_norm 40 --max_generation 80000 --goal_matrix_variance 1
python3 EvolTrajectory.py "S4FigCD/EvolTrajectoryA40_R6_VN" 6 --init_net_norm 40 --max_generation 150000 --goal_matrix_variance 1


#### S5Fig Analyzed data ####
mkdir S5Fig
python3 EvolTrajectory.py "S5Fig/EvolTrajectoryA001_R1_N12" 1 --init_net_norm 0.01 --max_generation 10000 --number_of_node 12
python3 EvolTrajectory.py "S5Fig/EvolTrajectoryA001_R2_N12" 2 --init_net_norm 0.01 --max_generation 15000 --number_of_node 12 
python3 EvolTrajectory.py "S5Fig/EvolTrajectoryA001_R3_N12" 3 --init_net_norm 0.01 --max_generation 80000 --number_of_node 12
python3 EvolTrajectory.py "S5Fig/EvolTrajectoryA001_R6_N12" 6 --init_net_norm 0.01 --max_generation 150000 --number_of_node 12
python3 EvolTrajectory.py "S5Fig/EvolTrajectoryA40_R1_N12" 1 --init_net_norm 40 --max_generation 10000 --number_of_node 12
python3 EvolTrajectory.py "S5Fig/EvolTrajectoryA40_R2_N12" 2 --init_net_norm 40 --max_generation 15000 --number_of_node 12
python3 EvolTrajectory.py "S5Fig/EvolTrajectoryA40_R3_N12" 3 --init_net_norm 40 --max_generation 80000 --number_of_node 12
python3 EvolTrajectory.py "S5Fig/EvolTrajectoryA40_R6_N12" 6 --init_net_norm 40 --max_generation 150000 --number_of_node 12


#### S2FigAB analyzed data ####
mkdir S2FigAB
python3 DynamicsInitDependency.py "S2FigAB/DynamInitDepend_R1" 1 --sampling_period_in_early_phase 10000
python3 DynamicsInitDependency.py "S2FigAB/DynamInitDepend_R2" 2 --sampling_period_in_early_phase 15000
python3 DynamicsInitDependency.py "S2FigAB/DynamInitDepend_R3" 3 --sampling_period_in_early_phase 80000
python3 DynamicsInitDependency.py "S2FigAB/DynamInitDepend_R6" 6 --sampling_period_in_early_phase 150000

#### S4FigBE analyzed data ####
mkdir S4FigBE
python3 DynamicsInitDependency.py "S4FigBE/DynamInitDepend_R1_VN" 1 --sampling_period_in_early_phase 10000 --goal_matrix_variance 1
python3 DynamicsInitDependency.py "S4FigBE/DynamInitDepend_R2_VN" 2 --sampling_period_in_early_phase 15000 --goal_matrix_variance 1
python3 DynamicsInitDependency.py "S4FigBE/DynamInitDepend_R3_VN" 3 --sampling_period_in_early_phase 80000 --goal_matrix_variance 1
python3 DynamicsInitDependency.py "S4FigBE/DynamInitDepend_R6_VN" 6 --sampling_period_in_early_phase 150000 --goal_matrix_variance 1


#### S6FigA Analyzed data ####
mkdir S6FigA
python3 RankDependency.py "S6FigA/ODE_RankDepend_R1_A0001" 1 --init_net_norm 0.001 --algorithm "GD"
python3 RankDependency.py "S6FigA/ODE_RankDepend_R2_A0001" 2 --init_net_norm 0.001 --algorithm "GD"
python3 RankDependency.py "S6FigA/ODE_RankDepend_R3_A0001" 3 --init_net_norm 0.001 --algorithm "GD"
python3 RankDependency.py "S6FigA/ODE_RankDepend_R6_A0001" 6 --init_net_norm 0.001 --algorithm "GD"


#### S6FigB Analyzed data ####
mkdir S6FigB
python3 InitDependency.py "S6FigB/ODE_InitDepend_R1" 1 --algorithm "GD"
python3 InitDependency.py "S6FigB/ODE_InitDepend_R2" 2 --algorithm "GD"
python3 InitDependency.py "S6FigB/ODE_InitDepend_R3" 3 --algorithm "GD"
python3 InitDependency.py "S6FigB/ODE_InitDepend_R6" 6 --algorithm "GD"

#### S6FigCD Analyzed data ####
mkdir S6FigCD
python3 EvolTrajectory.py "S6FigCD/ODE_EvolTrajectoryA0001_R1" 1 --init_net_norm 0.001 --max_generation 60000 --algorithm "GD"
python3 EvolTrajectory.py "S6FigCD/ODE_EvolTrajectoryA0001_R2" 2 --init_net_norm 0.001 --max_generation 60000 --algorithm "GD"
python3 EvolTrajectory.py "S6FigCD/ODE_EvolTrajectoryA0001_R3" 3 --init_net_norm 0.001 --max_generation 60000 --algorithm "GD"
python3 EvolTrajectory.py "S6FigCD/ODE_EvolTrajectoryA0001_R6" 6 --init_net_norm 0.001 --max_generation 1000000 --algorithm "GD"
python3 EvolTrajectory.py "S6FigCD/ODE_EvolTrajectoryA40_R1" 1 --init_net_norm 40 --max_generation 60000 --algorithm "GD"
python3 EvolTrajectory.py "S6FigCD/ODE_EvolTrajectoryA40_R2" 2 --init_net_norm 40 --max_generation 60000 --algorithm "GD"
python3 EvolTrajectory.py "S6FigCD/ODE_EvolTrajectoryA40_R3" 3 --init_net_norm 40 --max_generation 60000 --algorithm "GD"
python3 EvolTrajectory.py "S6FigCD/ODE_EvolTrajectoryA40_R6" 6 --init_net_norm 40 --max_generation 1000000 --algorithm "GD"


#### S1Fig Analyzed data ###
mkdir S1Fig
python3 RankDependency.py "S1FigA/RankDepend_R1_A40" 1 --init_net_norm 40
python3 RankDependency.py "S1FigA/RankDepend_R2_A40" 2 --init_net_norm 40
python3 RankDependency.py "S1FigA/RankDepend_R3_A40" 3 --init_net_norm 40 
python3 RankDependency.py "S1FigA/RankDepend_R6_A40" 6 --init_net_norm 40 

python3 RankDependency.py "S1FigA/RankDepend_R1_A40_EliteSl" 1 --init_net_norm 40 --selection_method "elite"
python3 RankDependency.py "S1FigA/RankDepend_R2_A40_EliteSl" 2 --init_net_norm 40 --selection_method "elite"
python3 RankDependency.py "S1FigA/RankDepend_R3_A40_EliteSl" 3 --init_net_norm 40 --selection_method "elite"
python3 RankDependency.py "S1FigA/RankDepend_R6_A40_EliteSl" 6 --init_net_norm 40 --selection_method "elite"

python3 RankDependency.py "S1FigA/RankDepend_R1_A001_EliteSl" 1 --init_net_norm 0.01 --selection_method "elite"
python3 RankDependency.py "S1FigA/RankDepend_R2_A001_EliteSl" 2 --init_net_norm 0.01 --selection_method "elite"
python3 RankDependency.py "S1FigA/RankDepend_R3_A001_EliteSl" 3 --init_net_norm 0.01 --selection_method "elite"
python3 RankDependency.py "S1FigA/RankDepend_R6_A001_EliteSl" 6 --init_net_norm 0.01 --selection_method "elite"

# plus Fig3A analyzed data

### S9 Fig Analyzed data ###
mkdir S9Fig
python3 RankDependency.py "S9FigA/RankDepend_R1_A40_L1" 1 --init_net_norm 40 --evaluation 1 --max_generation 10000 --skip_saturation_stop 1
python3 RankDependency.py "S9FigA/RankDepend_R6_A40_L1" 6 --init_net_norm 40 --evaluation 1 --max_generation 100000 --skip_saturation_stop 1
python3 RankDependency.py "S9FigA/RankDepend_R1_A40_L2" 1 --init_net_norm 40 --evaluation 2 --max_generation 10000 --skip_saturation_stop 1
python3 RankDependency.py "S9FigA/RankDepend_R6_A40_L2" 6 --init_net_norm 40 --evaluation 2 --max_generation 100000 --skip_saturation_stop 1


### Fig4 Analyzed data ###
mkdir Fig4
#A
python3 EvolTrajectory.py "Fig4/EvolTrajectoryA001_R1_AfterAdapt" 1 --init_net_norm 0.01 --max_generation 10000 --goal_fluctuation_mode 0
python3 EvolTrajectory.py "Fig4/EvolTrajectoryA001_R6_AfterAdapt" 6 --init_net_norm 0.01 --max_generation 200000 --goal_fluctuation_mode 0
#B
python3 EvolTrajectory.py "Fig4/EvolTrajectoryA001_R1_AfterAdaptRankChange" 1 --init_net_norm 0.01 --max_generation 200000 --goal_fluctuation_mode 1
#C
python3 EvolTrajectory.py "Fig4/EvolTrajectoryA001_R1_BeforeAdapt" 1 --init_net_norm 0.01 --max_generation 10000 --goal_fluctuation_mode 2
python3 EvolTrajectory.py "Fig4/EvolTrajectoryA001_R6_BeforeAdapt" 6 --init_net_norm 0.01 --max_generation 10000 --goal_fluctuation_mode 2


mkdir Fig5
python3 EvolTrajectory.py "Fig5/EvolTrajectoryA10_R1_Duplicate" 1 --init_net_norm 10 --max_generation 10000 --number_of_node 3 --number_of_layer 3 --goal_duplication 1
python3 EvolTrajectory.py "Fig5/EvolTrajectoryA10_R1_NoDuplicate" 1 --init_net_norm 10 --max_generation 10000 --number_of_node 3 --number_of_layer 3 --goal_duplication 0


