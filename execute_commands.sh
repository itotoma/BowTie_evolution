#!/bin/sh

# This shell file enumerates commands which generates data analyzed in the Itoh et al., 2023.
# Since each command takes time to complete, execution in high performance computer is preffered.

# Note
# Some scripts lost unneccesary information for generating corresponding figure. 
# If you want to obtain full information (e.g., evolutionary trajectory of network structure and fitness), please run DynamicsInitDependncy.py (GeneticAlgorithm function returns full information by setting output_style=2).

#### Fig3A  ####
mkdir Fig3A
python3 RankDependency.py "Fig3A/RankDepend_R1_A001" 1 --init_net_norm 0.01 
python3 RankDependency.py "Fig3A/RankDepend_R2_A001" 2 --init_net_norm 0.01
python3 RankDependency.py "Fig3A/RankDepend_R3_A001" 3 --init_net_norm 0.01
python3 RankDependency.py "Fig3A/RankDepend_R4_A001" 4 --init_net_norm 0.01
python3 RankDependency.py "Fig3A/RankDepend_R5_A001" 5 --init_net_norm 0.01 --max_generation 150000
python3 RankDependency.py "Fig3A/RankDepend_R6_A001" 6 --init_net_norm 0.01 --max_generation 150000


#### Fig3B  ####
mkdir Fig3B
python3 InitDependency.py "Fig3B/InitDepend_R1" 1
python3 InitDependency.py "Fig3B/InitDepend_R2" 2
python3 InitDependency.py "Fig3B/InitDepend_R3" 3
python3 InitDependency.py 'Fig3B/InitDepend_R4' 4
python3 InitDependency.py 'Fig3B/InitDepend_R5' 5
python3 InitDependency.py "Fig3B/InitDepend_R6" 6

#### S4FigA  ####
mkdir S4FigA
python3 InitDependency.py "S4FigA/InitDepend_R1_VN" 1 --goal_matrix_variance 1
python3 InitDependency.py "S4FigA/InitDepend_R2_VN" 2 --goal_matrix_variance 1
python3 InitDependency.py "S4FigA/InitDepend_R3_VN" 3 --goal_matrix_variance 1
python3 InitDependency.py "S4FigA/InitDepend_R6_VN" 6 --goal_matrix_variance 1


#### Fig3CD  ####
mkdir Fig3CD
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA001_R1" 1 --init_net_norm 0.01 --max_generation 10000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA001_R2" 2 --init_net_norm 0.01 --max_generation 15000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA001_R3" 3 --init_net_norm 0.01 --max_generation 80000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA001_R6" 4 --init_net_norm 0.01 --max_generation 80000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA001_R6" 5 --init_net_norm 0.01 --max_generation 150000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA001_R6" 6 --init_net_norm 0.01 --max_generation 150000

python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA40_R1" 1 --init_net_norm 40 --max_generation 10000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA40_R2" 2 --init_net_norm 40 --max_generation 15000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA40_R3" 3 --init_net_norm 40 --max_generation 80000 
ython3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA001_R6" 4 --init_net_norm 40 --max_generation 80000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA001_R6" 5 --init_net_norm 40 --max_generation 150000
python3 EvolTrajectory.py "Fig3CD/EvolTrajectoryA40_R6" 6 --init_net_norm 40 --max_generation 150000

#### SFig6 ab ####
mkdir AlternativeDef1_ab
python3 EvolTrajectory.py "AlternativeDef1_ab/EvolTrajectoryA001_R1_AN1" 1 --init_net_norm 0.01 --max_generation 10000 --active_node 1
python3 EvolTrajectory.py "AlternativeDef1_ab/EvolTrajectoryA001_R2_AN1" 2 --init_net_norm 0.01 --max_generation 15000 --active_node 1
python3 EvolTrajectory.py "AlternativeDef1_ab/EvolTrajectoryA001_R3_AN1" 3 --init_net_norm 0.01 --max_generation 80000 --active_node 1
python3 EvolTrajectory.py "AlternativeDef1_ab/EvolTrajectoryA001_R6_AN1" 6 --init_net_norm 0.01 --max_generation 150000 --active_node 1
python3 EvolTrajectory.py "AlternativeDef1_ab/EvolTrajectoryA40_R1_AN1" 1 --init_net_norm 40 --max_generation 10000 --active_node 1
python3 EvolTrajectory.py "AlternativeDef1_ab/EvolTrajectoryA40_R2_AN1" 2 --init_net_norm 40 --max_generation 15000 --active_node 1
python3 EvolTrajectory.py "AlternativeDef1_ab/EvolTrajectoryA40_R3_AN1" 3 --init_net_norm 40 --max_generation 80000 --active_node 1
python3 EvolTrajectory.py "AlternativeDef1_ab/EvolTrajectoryA40_R6_AN1" 6 --init_net_norm 40 --max_generation 150000 --active_node 1


#### SFig6 cd ####
mkdir AlternativeDef2_cd
python3 EvolTrajectory.py "AlternativeDef2_cd/EvolTrajectoryA001_R1_AN2" 1 --init_net_norm 0.01 --max_generation 10000 --active_node 2
python3 EvolTrajectory.py "AlternativeDef2_cd/EvolTrajectoryA001_R2_AN2" 2 --init_net_norm 0.01 --max_generation 15000 --active_node 2
python3 EvolTrajectory.py "AlternativeDef2_cd/EvolTrajectoryA001_R3_AN2" 3 --init_net_norm 0.01 --max_generation 80000 --active_node 2
python3 EvolTrajectory.py "AlternativeDef2_cd/EvolTrajectoryA001_R6_AN2" 6 --init_net_norm 0.01 --max_generation 150000 --active_node 2
python3 EvolTrajectory.py "AlternativeDef2_cd/EvolTrajectoryA40_R1_AN2" 1 --init_net_norm 40 --max_generation 10000 --active_node 2
python3 EvolTrajectory.py "AlternativeDef2_cd/EvolTrajectoryA40_R2_AN2" 2 --init_net_norm 40 --max_generation 15000 --active_node 2
python3 EvolTrajectory.py "AlternativeDef2_cd/EvolTrajectoryA40_R3_AN2" 3 --init_net_norm 40 --max_generation 80000 --active_node 2
python3 EvolTrajectory.py "AlternativeDef2_cd/EvolTrajectoryA40_R6_AN2" 6 --init_net_norm 40 --max_generation 150000 --active_node 2


#### SFig7 Analyzed data ####
mkdir VarianceNorm_cd
python3 EvolTrajectory.py "VarianceNorm_cd/EvolTrajectoryA001_R1_VN" 1 --init_net_norm 0.01 --max_generation 10000 --goal_matrix_variance 1
python3 EvolTrajectory.py "VarianceNorm_cd/EvolTrajectoryA001_R2_VN" 2 --init_net_norm 0.01 --max_generation 15000 --goal_matrix_variance 1
python3 EvolTrajectory.py "VarianceNorm_cd/EvolTrajectoryA001_R3_VN" 3 --init_net_norm 0.01 --max_generation 80000 --goal_matrix_variance 1
python3 EvolTrajectory.py "VarianceNorm_cd/EvolTrajectoryA001_R6_VN" 6 --init_net_norm 0.01 --max_generation 150000 --goal_matrix_variance 1
python3 EvolTrajectory.py "VarianceNorm_cd/EvolTrajectoryA40_R1_VN" 1 --init_net_norm 40 --max_generation 10000 --goal_matrix_variance 1
python3 EvolTrajectory.py "VarianceNorm_cd/EvolTrajectoryA40_R2_VN" 2 --init_net_norm 40 --max_generation 15000 --goal_matrix_variance 1
python3 EvolTrajectory.py "VarianceNorm_cd/EvolTrajectoryA40_R3_VN" 3 --init_net_norm 40 --max_generation 80000 --goal_matrix_variance 1
python3 EvolTrajectory.py "VarianceNorm_cd/EvolTrajectoryA40_R6_VN" 6 --init_net_norm 40 --max_generation 150000 --goal_matrix_variance 1


#### S5Fig Analyzed data ####
mkdir Node12Network
python3 EvolTrajectory.py "Node12Network/EvolTrajectoryA001_R1_N12" 1 --init_net_norm 0.01 --max_generation 10000 --number_of_node 12
python3 EvolTrajectory.py "Node12Network/EvolTrajectoryA001_R2_N12" 2 --init_net_norm 0.01 --max_generation 15000 --number_of_node 12 
python3 EvolTrajectory.py "Node12Network/EvolTrajectoryA001_R3_N12" 3 --init_net_norm 0.01 --max_generation 80000 --number_of_node 12
python3 EvolTrajectory.py "Node12Network/EvolTrajectoryA001_R6_N12" 6 --init_net_norm 0.01 --max_generation 150000 --number_of_node 12
python3 EvolTrajectory.py "Node12Network/EvolTrajectoryA40_R1_N12" 1 --init_net_norm 40 --max_generation 10000 --number_of_node 12
python3 EvolTrajectory.py "Node12Network/EvolTrajectoryA40_R2_N12" 2 --init_net_norm 40 --max_generation 15000 --number_of_node 12
python3 EvolTrajectory.py "Node12Network/EvolTrajectoryA40_R3_N12" 3 --init_net_norm 40 --max_generation 80000 --number_of_node 12
python3 EvolTrajectory.py "Node12Network/EvolTrajectoryA40_R6_N12" 6 --init_net_norm 40 --max_generation 150000 --number_of_node 12


#### SFig 3a analyzed data ####
mkdir DynamicsInitDepend
#DynamicInitDependency.py returns 3 result file.
python3 DynamicsInitDependency.py "DynamicsInitDepend/DynamInitDepend_R1" 1 --sampling_period_in_early_phase 10000
python3 DynamicsInitDependency.py "DynamicsInitDepend/DynamInitDepend_R2" 2 --sampling_period_in_early_phase 15000
python3 DynamicsInitDependency.py "DynamicsInitDepend/DynamInitDepend_R3" 3 --sampling_period_in_early_phase 80000
#python3 DynamicsInitDependency.py "DynamicsInitDepend/DynamInitDepend_R6" 6 --sampling_period_in_early_phase 150000

#### SFig 7be analyzed data ####
mkdir VarianceNorm_be
python3 DynamicsInitDependency.py "VarianceNorm_be/DynamInitDepend_R1" 1 --sampling_period_in_early_phase 10000 --goal_matrix_variance 1
python3 DynamicsInitDependency.py "VarianceNorm_be/DynamInitDepend_R2" 2 --sampling_period_in_early_phase 15000 --goal_matrix_variance 1
python3 DynamicsInitDependency.py "VarianceNorm_be/DynamInitDepend_R3" 3 --sampling_period_in_early_phase 80000 --goal_matrix_variance 1
python3 DynamicsInitDependency.py "VarianceNorm_be/DynamInitDepend_R6" 6 --sampling_period_in_early_phase 150000 --goal_matrix_variance 1


#### SFig 8a Analyzed data ####
mkdir ODE_a
python3 RankDependency.py "ODE_a/ODE_RankDepend_R1_A0001" 1 --init_net_norm 0.001 --algorithm "GD"
python3 RankDependency.py "ODE_a/ODE_RankDepend_R2_A0001" 2 --init_net_norm 0.001 --algorithm "GD"
python3 RankDependency.py "ODE_a/ODE_RankDepend_R3_A0001" 3 --init_net_norm 0.001 --algorithm "GD"
python3 RankDependency.py "ODE_a/ODE_RankDepend_R6_A0001" 6 --init_net_norm 0.001 --algorithm "GD"


#### SFig 8b Analyzed data ####
mkdir ODE_b
python3 InitDependency.py "ODE_b/ODE_InitDepend_R1" 1 --algorithm "GD"
python3 InitDependency.py "ODE_b/ODE_InitDepend_R2" 2 --algorithm "GD"
python3 InitDependency.py "ODE_b/ODE_InitDepend_R3" 3 --algorithm "GD"
python3 InitDependency.py "ODE_b/ODE_InitDepend_R6" 6 --algorithm "GD"

#### SFig 8cd Analyzed data ####
mkdir ODE_cd
python3 EvolTrajectory.py "ODE_cd/ODE_EvolTrajectoryA0001_R1" 1 --init_net_norm 0.001 --max_generation 60000 --algorithm "GD"
python3 EvolTrajectory.py "ODE_cd/ODE_EvolTrajectoryA0001_R2" 2 --init_net_norm 0.001 --max_generation 60000 --algorithm "GD"
python3 EvolTrajectory.py "ODE_cd/ODE_EvolTrajectoryA0001_R3" 3 --init_net_norm 0.001 --max_generation 60000 --algorithm "GD"
python3 EvolTrajectory.py "ODE_cd/ODE_EvolTrajectoryA0001_R6" 6 --init_net_norm 0.001 --max_generation 1000000 --algorithm "GD"
python3 EvolTrajectory.py "ODE_cd/ODE_EvolTrajectoryA40_R1" 1 --init_net_norm 40 --max_generation 60000 --algorithm "GD"
python3 EvolTrajectory.py "ODE_cd/ODE_EvolTrajectoryA40_R2" 2 --init_net_norm 40 --max_generation 60000 --algorithm "GD"
python3 EvolTrajectory.py "ODE_cd/ODE_EvolTrajectoryA40_R3" 3 --init_net_norm 40 --max_generation 60000 --algorithm "GD"
python3 EvolTrajectory.py "ODE_cd/ODE_EvolTrajectoryA40_R6" 6 --init_net_norm 40 --max_generation 1000000 --algorithm "GD"


#### SFig 2 Analyzed data ###
mkdir VariousSelections
python3 RankDependency.py "VariousSelections/RankDepend_R1_A40" 1 --init_net_norm 40
python3 RankDependency.py "VariousSelections/RankDepend_R2_A40" 2 --init_net_norm 40
python3 RankDependency.py "VariousSelections/RankDepend_R3_A40" 3 --init_net_norm 40 
python3 RankDependency.py "VariousSelections/RankDepend_R6_A40" 6 --init_net_norm 40 

python3 RankDependency.py "VariousSelections/RankDepend_R1_A40_EliteSl" 1 --init_net_norm 40 --selection_method "elite"
python3 RankDependency.py "VariousSelections/RankDepend_R2_A40_EliteSl" 2 --init_net_norm 40 --selection_method "elite"
python3 RankDependency.py "VariousSelections/RankDepend_R3_A40_EliteSl" 3 --init_net_norm 40 --selection_method "elite"
python3 RankDependency.py "VariousSelections/RankDepend_R6_A40_EliteSl" 6 --init_net_norm 40 --selection_method "elite"

python3 RankDependency.py "VariousSelections/RankDepend_R1_A001_EliteSl" 1 --init_net_norm 0.01 --selection_method "elite"
python3 RankDependency.py "VariousSelections/RankDepend_R2_A001_EliteSl" 2 --init_net_norm 0.01 --selection_method "elite"
python3 RankDependency.py "VariousSelections/RankDepend_R3_A001_EliteSl" 3 --init_net_norm 0.01 --selection_method "elite"
python3 RankDependency.py "VariousSelections/RankDepend_R6_A001_EliteSl" 6 --init_net_norm 0.01 --selection_method "elite"


### SFig 12 ###
mkdir Regularizations
python3 RankDependency.py "Regularizations/RankDepend_R1_A40_L1" 1 --init_net_norm 40 --evaluation 1 --max_generation 10000 --skip_saturation_stop 1
python3 RankDependency.py "Regularizations/RankDepend_R6_A40_L1" 6 --init_net_norm 40 --evaluation 1 --max_generation 100000 --skip_saturation_stop 1
python3 RankDependency.py "Regularizations/RankDepend_R1_A40_L2" 1 --init_net_norm 40 --evaluation 2 --max_generation 10000 --skip_saturation_stop 1
python3 RankDependency.py "Regularizations/RankDepend_R6_A40_L2" 6 --init_net_norm 40 --evaluation 2 --max_generation 100000 --skip_saturation_stop 1


### Fig4 Analyzed data ###
mkdir Fig4
#a
python3 EvolTrajectory.py "Fig4/EvolTrajectoryA001_R1_AfterAdapt" 1 --init_net_norm 0.01 --max_generation 10000 --goal_fluctuation_mode 0
python3 EvolTrajectory.py "Fig4/EvolTrajectoryA001_R6_AfterAdapt" 6 --init_net_norm 0.01 --max_generation 200000 --goal_fluctuation_mode 0
#b
python3 EvolTrajectory.py "Fig4/EvolTrajectoryA001_R1_AfterAdaptRankChange" 1 --init_net_norm 0.01 --max_generation 200000 --goal_fluctuation_mode 1
#c
python3 EvolTrajectory.py "Fig4/EvolTrajectoryA001_R1_BeforeAdapt" 1 --init_net_norm 0.01 --max_generation 10000 --goal_fluctuation_mode 2
python3 EvolTrajectory.py "Fig4/EvolTrajectoryA001_R6_BeforeAdapt" 6 --init_net_norm 0.01 --max_generation 10000 --goal_fluctuation_mode 2


mkdir Fig5
python3 EvolTrajectory.py "Fig5/Fig5R10N10L5_DU" 10 --init_net_norm 10 --max_generation 10000 --number_of_node 10 --number_of_layer 5 --goal_duplication 1
python3 EvolTrajectory.py "Fig5/Fig5R10N10L5_ND" 10 --init_net_norm 10 --max_generation 10000 --number_of_node 10 --number_of_layer 5 --goal_duplication 0





### SFig 13 ###

for title in {0..9}
do
   python3 GeneticAlgorithm_NonLinear.py 0.01 'A0_001_$title'
done

for title in {0..9} 
do
    python3 GeneticAlgorithm_NonLinear.py 1000 'A0_1000_$title'
done


