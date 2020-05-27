from tests import generateRuns, test, testPolicy
from BlocksWorld import *
from MonteCarlo import *

generateRuns()
test()

# blocksWorld = BlocksWorld()

# mc = MonteCarlo(blocksWorld)
# learnedPolicy = mc.learnPolicy(maxEpisodeLength=8, gamma=1, numberEpisodes=200, planningFactor=1) # {state : action}
# print('Learned policy: %s' % learnedPolicy)
# print()

# testPolicy(learnedPolicy, mc, 8)
