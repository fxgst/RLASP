from tests import *
from BlocksWorld import *
from MonteCarlo import *

### generate test data ##########################
#generateRuns()

### generate plots from testdata ################
#plotMultiple('plot_5000e_20r_0pf', [('4b_5000e_20r_0pf', '4 blocks'), ('5b_5000e_20r_0pf', '5 blocks'), ('6b_5000e_20r_0pf', '6 blocks'), ('7b_5000e_20r_0pf', '7 blocks')])
#plotMultiple('plot_7b_2000e_40r_0pf_poep', [('7b_2000e_40r_0pf_poep_3phz', 'pH = 3'), ('7b_2000e_40r_0pf_poep_4phz', 'pH = 4'), ('7b_2000e_40r_0pf_poep_5phz', 'pH = 5'), ('7b_2000e_40r_0pf_poep_6phz', 'pH = 6')])
#plotMultiple('plot_7b_2000e_40r_5phz', [('7b_2000e_40r_0.4pf_5phz', 'pF = 0.4'), ('7b_2000e_40r_0.5pf_5phz', 'pF = 0.5'), ('7b_2000e_40r_0.6pf_5phz', 'pF = 0.6'), ('7b_2000e_40r_0.7pf_5phz', 'pF = 0.7')])


plotMultiple('plot_Nb_2000e_20r_0pf_poep_N-1phz', [ ('8b_2000e_20r_0pf_poep_7phz', '8b, pH=7'), ('9b_2000e_20r_0pf_poep_8phz', '9b, pH=8'), ('10b_2000e_20r_0pf_poep_9phz', '10b, pH=9'), ('12b_2000e_20r_0pf_poep_11phz', '12b, pH=11')])
#('7b_2000e_40r_0pf_poep_6phz', '7b, pH=6'),, ('11b_2000e_20r_0pf_poep_10phz', '11b, pH=10')

#plotMultiple('plot_5000e_10r_0pf', [('4b_5000e_10r_0pf', '4 blocks'), ('5b_5000e_10r_0pf', '5 blocks'), ('6b_5000e_10r_0pf', '6 blocks'), ('7b_5000e_10r_0pf', '7 blocks')])

### testing #####################################
# blocksWorld = BlocksWorld()
# mc = MonteCarlo(blocksWorld)
# learnedPolicy = mc.learnPolicy(maxEpisodeLength=12, gamma=1, numberEpisodes=100, planning_factor=0, plan_on_empty_policy=False, planning_horizon=0) # {state : action}
# print('Learned policy: %s' % learnedPolicy)
# print()

# testPolicy(learnedPolicy, mc, 12)
