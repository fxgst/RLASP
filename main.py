from tests import *
from MonteCarlo import *

blocks_world = BlocksWorld()
mc = MonteCarlo(blocks_world, max_episode_length=8, planning_factor=0, plan_on_empty_policy=False, planning_horizon=0)
learned_policy = mc.learn_policy(gamma=1, number_episodes=150)  # {state : action}
print(f'Learned policy: {learned_policy}')
print()

test_policy(learned_policy, blocks_world, max_episode_length=8)
