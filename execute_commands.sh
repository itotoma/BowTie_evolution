#!/bin/sh

# This shell file enumerates commands which generates data analyzed in the Itoh et al., 2023.
# Since each command takes time to complete, execution in high performance computer is preffered.

# Note
# Each scripts lost unneccesary information for generating corresponding figure 
# If you want to grasp full information, execute XXXXXX

#### Fig3A Analyzed data ####
# mkdir RidM1
# python3 RankDependency.py 1 "RidM1/N100_RankDepend_R1_A001" GA 0.01
# python3 RankDependency.py 2 "RidM1/N100_RankDepend_R2_A001" GA 0.01
# python3 RankDependency.py 3 "RidM1/N100_RankDepend_R3_A001" GA 0.01
# python3 RankDependency.py 6 "RidM1/N100_RankDepend_R6_A001" GA 0.01

#### Fig3B Analyzed data ####
# mkdir RidM3
# parameter: GA_InitDependency.py {Goal matrix rank} {output path} {variance normalize}
# python3 InitDependency.py 1 "RidM3/InitDepend_R1" 0 GA
# python3 InitDependency.py 2 "RidM3/InitDepend_R2" 0 GA
# python3 InitDependency.py 3 "RidM3/InitDepend_R3" 0 GA
# python3 InitDependency.py 6 "RidM3/InitDepend_R6" 0 GA

#### S4FigA Analyzed data ####
# mkdir Rid5
# parameter:_InitDependency.py {Goal matrix rank} {output path} {variance normalize}}
# python3 InitDependency.py 1 "RidS5/InitDepend_R1_VN" 1
# python3 InitDependency.py 2 "RidS5/InitDepend_R2_VN" 1
# python3 InitDependency.py 3 "RidS5/InitDepend_R3_VN" 1
# python3 InitDependency.py 6 "RidS5/InitDepend_R6_VN" 1


#### Fig3CD Analyzed data ####
# mkdir RidM2
# python3 GA_TimeTrajectory.py 1 "RidM2/TimeTrajectoryA001_R1" 0.01 10000 0 0
# python3 GA_TimeTrajectory.py 2 "RidM2/TimeTrajectoryA001_R2" 0.01 15000 0 0
# python3 GA_TimeTrajectory.py 3 "RidM2/TimeTrajectoryA001_R3" 0.01 80000 0 0
# python3 GA_TimeTrajectory.py 6 "RidM2/TimeTrajectoryA001_R6" 0.01 150000 0 0
# python3 GA_TimeTrajectory.py 1 "RidM2/TimeTrajectoryA40_R1" 40 10000 0 0
# python3 GA_TimeTrajectory.py 2 "RidM2/TimeTrajectoryA40_R2" 40 15000 0 0
# python3 GA_TimeTrajectory.py 3 "RidM2/TimeTrajectoryA40_R3" 40 80000 0 0
# python3 GA_TimeTrajectory.py 6 "RidM2/TimeTrajectoryA40_R6" 40 150000 0 0

#### S3FigAB Analyzed data ####
# mkdir Rid1
# python3 GA_TimeTrajectory.py 1 "Rid1/TimeTrajectoryA001_R1_AN1" 0.01 10000 1 0
# python3 GA_TimeTrajectory.py 2 "Rid1/TimeTrajectoryA001_R2_AN1" 0.01 15000 1 0
# python3 GA_TimeTrajectory.py 3 "Rid1/TimeTrajectoryA001_R3_AN1" 0.01 80000 1 0
# python3 GA_TimeTrajectory.py 6 "Rid1/TimeTrajectoryA001_R6_AN1" 0.01 150000 1 0
# python3 GA_TimeTrajectory.py 1 "Rid1/TimeTrajectoryA40_R1_AN1" 40 10000 1 0 
# python3 GA_TimeTrajectory.py 2 "Rid1/TimeTrajectoryA40_R2_AN1" 40 15000 1 0 
# python3 GA_TimeTrajectory.py 3 "Rid1/TimeTrajectoryA40_R3_AN1" 40 80000 1 0 
# python3 GA_TimeTrajectory.py 6 "Rid1/TimeTrajectoryA40_R6_AN1" 40 150000 1 0


#### S3FigCD Analyzed data ####
# mkdir Rid2
# python3 GA_TimeTrajectory.py 1 "Rid2/TimeTrajectoryA001_R1_AN2" 0.01 10000 2 0
# python3 GA_TimeTrajectory.py 2 "Rid2/TimeTrajectoryA001_R2_AN2" 0.01 15000 2 0
# python3 GA_TimeTrajectory.py 3 "Rid2/TimeTrajectoryA001_R3_AN2" 0.01 80000 2 0
# python3 GA_TimeTrajectory.py 6 "Rid2/TimeTrajectoryA001_R6_AN2" 0.01 150000 2 0
# python3 GA_TimeTrajectory.py 1 "Rid2/TimeTrajectoryA40_R1_AN2" 40 10000 2 0
# python3 GA_TimeTrajectory.py 2 "Rid2/TimeTrajectoryA40_R2_AN2" 40 15000 2 0
# python3 GA_TimeTrajectory.py 3 "Rid2/TimeTrajectoryA40_R3_AN2" 40 80000 2 0
# python3 GA_TimeTrajectory.py 6 "Rid2/TimeTrajectoryA40_R6_AN2" 40 150000 2 0


#### S4FigCD Analyzed data ####
# mkdir Rid3
# python3 GA_TimeTrajectory.py 1 "Rid3/TimeTrajectoryA001_R1_VN" 0.01 10000 0 1
# python3 GA_TimeTrajectory.py 2 "Rid3/TimeTrajectoryA001_R2_VN" 0.01 15000 0 1
# python3 GA_TimeTrajectory.py 3 "Rid3/TimeTrajectoryA001_R3_VN" 0.01 80000 0 1
# python3 GA_TimeTrajectory.py 6 "Rid3/TimeTrajectoryA001_R6_VN" 0.01 150000 0 1
# python3 GA_TimeTrajectory.py 1 "Rid3/TimeTrajectoryA40_R1_VN" 40 10000 0 1
# python3 GA_TimeTrajectory.py 2 "Rid3/TimeTrajectoryA40_R2_VN" 40 15000 0 1
# python3 GA_TimeTrajectory.py 3 "Rid3/TimeTrajectoryA40_R3_VN" 40 80000 0 1
# python3 GA_TimeTrajectory.py 6 "Rid3/TimeTrajectoryA40_R6_VN" 40 150000 0 1


#### S5Fig Analyzed data ####
# mkdir Rid8
# python3 GA_TimeTrajectory.py 1 "Rid8/TimeTrajectoryA001_R1_N12" 0.01 10000 0 0 12
# python3 GA_TimeTrajectory.py 2 "Rid8/TimeTrajectoryA001_R2_N12" 0.01 15000 0 0 12
# python3 GA_TimeTrajectory.py 3 "Rid8/TimeTrajectoryA001_R3_N12" 0.01 80000 0 0 12
# python3 GA_TimeTrajectory.py 6 "Rid8/TimeTrajectoryA001_R6_N12" 0.01 150000 0 0 12
# python3 GA_TimeTrajectory.py 1 "Rid8/TimeTrajectoryA40_R1_N12" 40 10000 0 0 12
# python3 GA_TimeTrajectory.py 2 "Rid8/TimeTrajectoryA40_R2_N12" 40 15000 0 0 12
# python3 GA_TimeTrajectory.py 3 "Rid8/TimeTrajectoryA40_R3_N12" 40 80000 0 0 12
# python3 GA_TimeTrajectory.py 6 "Rid8/TimeTrajectoryA40_R6_N12" 40 150000 0 0 12


#### S2FigAB analyzed data ####
# parameter: {Goal matrix rank} {output path} {Sampling period in early phase} {Goal variance normalize}
# mkdir Rid6
# python3 GA_GeneralDependency.py 1 "Rid6/GeneDep_R1" 10000 0
# python3 GA_GeneralDependency.py 2 "Rid6/GeneDep_R2" 15000 0
# python3 GA_GeneralDependency.py 3 "Rid6/GeneDep_R3" 80000 0
# python3 GA_GeneralDependency.py 6 "Rid6/GeneDep_R6" 150000 0

#### S4FigBE analyzed data ####
# parameter: {Goal matrix rank} {output path} {Sampling period in early phase} {Goal variance normalize}
# mkdir Rid6VN
# python3 GA_GeneralDependency.py 1 "Rid6VN/GeneDep_R1_VN" 10000 1
# python3 GA_GeneralDependency.py 2 "Rid6VN/GeneDep_R2_VN" 15000 1
# python3 GA_GeneralDependency.py 3 "Rid6VN/GeneDep_R3_VN" 80000 1
# python3 GA_GeneralDependency.py 6 "Rid6VN/GeneDep_R6_VN" 150000 1


#### S6FigA Analyzed data ####
# mkdir S6FigA
# python3 RankDependency.py 1 "S6FigA/N100_RankDepend_R1_A001" GD 0.001
# python3 RankDependency.py 2 "S6FigA/N100_RankDepend_R2_A001" GD 0.001
# python3 RankDependency.py 3 "S6FigA/N100_RankDepend_R3_A001" GD 0.001
# python3 RankDependency.py 6 "S6FigA/N100_RankDepend_R6_A001" GD 0.001


#### S6FigB Analyzed data ####
# mkdir RidM3
# parameter: GA_InitDependency.py {Goal matrix rank} {output path} {variance normalize}
# python3 InitDependency.py 1 "RidM3/InitDepend_R1" 0 GD
# python3 InitDependency.py 2 "RidM3/InitDepend_R2" 0 GD
# python3 InitDependency.py 3 "RidM3/InitDepend_R3" 0 GD
# python3 InitDependency.py 6 "RidM3/InitDepend_R6" 0 GD

#### S6FigC Analyzed data ####
# mkdir RidM2
# python3 TimeTrajectory.py 1 "RidM2/TimeTrajectoryA001_R1" 0.01 #### Fig3CD Analyzed data ####
# mkdir RidM2
# python3 GA_TimeTrajectory.py 1 "RidM2/TimeTrajectoryA001_R1" 0.001 60000 0 0 0 GD
# python3 GA_TimeTrajectory.py 2 "RidM2/TimeTrajectoryA001_R2" 0.001 60000 0 0 0 GD
# python3 GA_TimeTrajectory.py 3 "RidM2/TimeTrajectoryA001_R3" 0.001 60000 0 0 0 GD
# python3 GA_TimeTrajectory.py 6 "RidM2/TimeTrajectoryA001_R6" 0.001 1000000 0 0 0 GD
# python3 GA_TimeTrajectory.py 1 "RidM2/TimeTrajectoryA40_R1" 40 60000 0 0 0 GD
# python3 GA_TimeTrajectory.py 2 "RidM2/TimeTrajectoryA40_R2" 40 60000 0 0 0 GD
# python3 GA_TimeTrajectory.py 3 "RidM2/TimeTrajectoryA40_R3" 40 60000 0 0 0 GD
# python3 GA_TimeTrajectory.py 6 "RidM2/TimeTrajectoryA40_R6" 40 1000000 0 0 0 GD
