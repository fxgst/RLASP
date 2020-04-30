# Blocksworld
This program combines reinforcement learning (RL) with answer set programming (ASP) to evaluate a policy for the blocks world problem.
The technique used for reinforcement learning is a first visit exploring starts Monte Carlo policy evaluation method.
The main learning algorithm is a Python implementation of the pseudo code presented in this [book][rl_book] in chapter 5.3. 
For the ASP parts, like executing actions and determining the current state of the environment, `clingo` by [Potassco, the Potsdam Answer Set Solving Collection][potassco] is used.

## Usage
Run `main.py` with `python3`. Note that this program requires the `clingo` and `numpy` modules.
A guide on how to install the `clingo` Python API can be found [here][clingo_python_api].
Hyperparameters can be set in `main.py`, specifically at the `learnPolicy` method call.
You can add as many blocks and define the goal state as you wish, but performance might significantly decrease as more blocks are used.

[rl_book]: http://incompleteideas.net/book/the-book-2nd.html
[potassco]: https://potassco.org
[clingo_python_api]: https://potassco.org/clingo/#packages
