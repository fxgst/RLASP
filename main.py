from BlocksWorld import *
from MonteCarlo import *
from entities import State, PartState

goal = State({PartState('a,table'), PartState('c,b'), PartState('b,a')})
blocksWorld = BlocksWorld(['a', 'b', 'c'], goal)

mc = MonteCarlo(blocksWorld)
learnedPolicy = mc.learnPolicy(maxEpisodeLength=10, gamma=1, episodes=256) # {state : action}
print('Learned policy: %s' % learnedPolicy)
print()

# check whether goal can be reached from all start states
def testPolicy(maxSteps=10):
    for state in mc.allStates:
        (_, reward, _) = mc.generateEpisode(state, learnedPolicy, maxSteps, False).pop() # final step
        if reward == 0:
            print(state.clingoString() + '\t\t\t✅')
        else:
            print(state.clingoString() + '\t\t\t❌')
        
testPolicy()
