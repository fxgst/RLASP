from BlocksWorld import *
from MonteCarlo import *
from tests import testPolicy

blocksWorld = BlocksWorld()

mc = MonteCarlo(blocksWorld)
learnedPolicy = mc.learnPolicy(maxEpisodeLength=10, gamma=1, numberEpisodes=128) # {state : action}
print('Learned policy: %s' % learnedPolicy)
print()

testPolicy(learnedPolicy, mc)
