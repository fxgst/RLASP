from tests import *
from BlocksWorld import *
from MonteCarlo import *

### generate test data ##########################
# generate_runs()

### generate plots from testdata ################
# plot_multiple('plot_5000e_20r_0pf', [('4b_5000e_20r_0pf', '4 blocks'), ('5b_5000e_20r_0pf', '5 blocks'), ('6b_5000e_20r_0pf', '6 blocks'), ('7b_5000e_20r_0pf', '7 blocks')])
# plot_multiple('plot_7b_2000e_40r_0pf_poep', [('7b_2000e_40r_0pf_poep_3phz', 'pH = 3'), ('7b_2000e_40r_0pf_poep_4phz', 'pH = 4'), ('7b_2000e_40r_0pf_poep_5phz', 'pH = 5'), ('7b_2000e_40r_0pf_poep_6phz', 'pH = 6')])
# plot_multiple('plot_7b_2000e_40r_5phz', [('7b_2000e_40r_0.4pf_5phz', 'pF = 0.4'), ('7b_2000e_40r_0.5pf_5phz', 'pF = 0.5'), ('7b_2000e_40r_0.6pf_5phz', 'pF = 0.6'), ('7b_2000e_40r_0.7pf_5phz', 'pF = 0.7')])
# plot_multiple('plot_Nb_2000e_20r_0pf_poep_N-1phz', [('8b_2000e_20r_0pf_poep_7phz', '8b, pH=7'), ('9b_2000e_20r_0pf_poep_8phz', '9b, pH=8'), ('10b_2000e_20r_0pf_poep_9phz', '10b, pH=9'), ('12b_2000e_20r_0pf_poep_11phz', '12b, pH=11')])

### testing #####################################
blocks_world = BlocksWorld()
mc = MonteCarlo(blocks_world, max_episode_length=8, planning_factor=0, plan_on_empty_policy=False, planning_horizon=0)
learned_policy = mc.learn_policy(gamma=1, number_episodes=100) # {state : action}
print('Learned policy: %s' % learned_policy)
print()

test_policy(learned_policy, blocks_world, 8)
