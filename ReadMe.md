# Emergence of bow-tie architecture in evolving feedforward network

- ### All simulation commands and conditions are enumerated in the `execute_commands.sh`
- ### All data are uploaded in the zenodo https://zenodo.org/records/10207353
- ### All figure creation code are written in the `FigureGenerator.ipynb`



### `FigureGenerator.ipynb`
>Figure creation code.
>Data is downloded from zenodo in this script.

### `execute_commands.sh`
>This shell file enumerates commands which generate data used in the figure.
>You can see the detail of parameters by `python XXXX.py -h` .


#### `BasicFunctios.py`
>Set of functios. Genetic algorithm and gradient descent algorithm are also defined in this script.
>
#### `NetworkEvolution_LinearGA.ipynb`
>Genetic algorithm for lienar network evolution.
>This sript can be used for the test run of evolutionary simulation.

#### `NetworkEvolution_NonLinearGA.ipynb`
>Genetic algorithm for non linear network evolution.
>This sript can be used for the test run of evolutionary simulation.

#### `NetworkEvolution_ODE.ipynb`
>Gradient descnet for network evolution.
>This sript can be used for the test run of deterministic model.

#### `RankDependency.py`
>Simulation code which returns data used for the Fig3A format (Rank vs #node in each layer).
You can see parameters by `python RankDependency.py -h`

#### `InitDependency.py`
>Simulation code which returns data used for the Fig3B format (Initial value vs #node in a waist).
You can see parameters by `python InitDependency.py -h`

#### `EvolTrajectory.py`
>Simulation code which returns data used for the Fig3C format (Evolution trajectory of #node).
You can see parameters by `python EvolTrajectory.py -h`

#### `DynamicsInitDependency.py`
>Simulation code which returns data used for the SFig2 format (Probability for being bow-tie and initial dependency of instantaneous minimum wist size).
All scripts except `DynamicsInitDependency.py` lost unneccesary information for generating corresponding figure.
If you want to analyse detail dynamics, execution in `DynamicsInitDependency.py` is preffered.
You can see parameters by `python DynamicsInitDependency.py -h`

