from BlocksWorld import *
from MonteCarlo import *

blocksWorld = BlocksWorld()

mc = MonteCarlo(blocksWorld)
learnedPolicy = mc.learnPolicy(maxEpisodeLength=10, gamma=1, numberEpisodes=256) # {state : action}
print('Learned policy: %s' % learnedPolicy)
print()

# check whether goal can be reached from all start states
def testPolicy(maxSteps=10):
    for state in mc.allStates:
        steps = mc.generateEpisode(state, learnedPolicy, maxSteps, False, True)
        if steps:
            (_, reward, _) = steps.pop() # final step
            if reward == 99:
                print(f'{str(state):<100} {"✅":>1}')
            else:
                print(f'{str(state):<100} {"❌":>1}')
        else:
            print(f'{str(state):<100} {"✅":>1}') # empty episode means start == goal

testPolicy()
