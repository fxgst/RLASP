from tests import generateRuns, generatePlots, testPolicy
from BlocksWorld import *
from MonteCarlo import *

generateRuns()
generatePlots()

# blocksWorld = BlocksWorld()
# mc = MonteCarlo(blocksWorld)
# learnedPolicy = mc.learnPolicy(maxEpisodeLength=10, gamma=1, numberEpisodes=100, planningFactor=0, planOnEmptyPolicy=True, planningHorizon=10) # {state : action}
# print('Learned policy: %s' % learnedPolicy)
# print()

# testPolicy(learnedPolicy, mc, 10)
