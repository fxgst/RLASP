# RLASP
This program combines reinforcement learning (RL) with answer set programming (ASP) to evaluate policies for solving blocks world problems.
The technique used for reinforcement learning is a first-visit Exploring Starts Monte Carlo policy evaluation method.
For the ASP parts, like executing actions and determining the current state of the environment, `clingo` by [Potassco, the Potsdam Answer Set Solving Collection][potassco] is used.
ASP is also used to generate plans for accelerating the learning process.

## Usage
Run `main.py` with `python3`. Note that this program requires the `clingo` and `numpy` modules.
You may want to use a `conda` [environment][conda].
A guide on how to install the `clingo` Python API can be found [here][clingo_python_api].
Hyperparameters can be set in `main.py`, specifically at the `mc` variable.
You can add as many blocks and define the goal state as you wish in `*.lp`, but performance might decrease as more blocks are used.

## Reference
To read the full reference and theoretical backgrounds, take a look at and download the [thesis][thesis].

[potassco]: https://potassco.org
[clingo_python_api]: https://potassco.org/clingo/#packages
[conda]: https://docs.conda.io/en/latest/
[thesis]: https://fuxgeist.com/thesis.pdf
