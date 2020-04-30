from BlocksWorld import *
from MonteCarlo import *
from entities import Block, SubGoal

a = Block('a')
b = Block('b')
c = Block('c')
d = Block('d')
e = Block('e')
f = Block('f')
g = Block('g')
table = Block('table')

#blocks = {a, b}
blocks = {a, b, c}
#blocks = {a, b, c, d, e}

#goal = State({SubGoal(a, table), SubGoal(b, a)})
goal = State({SubGoal(a, table), SubGoal(b, a), SubGoal(c, b)})
#goal = State({SubGoal(a, table), SubGoal(b, a), SubGoal(c, b), SubGoal(d, c), SubGoal(e, d)})

blocksWorld = BlocksWorld(blocks, goal)

mc = MonteCarlo(blocksWorld)
learnedPolicy = mc.learnPolicy(maxEpisodeLength=20, gamma=1, episodes=300) # {state : (action, reward)}
print('Learned policy: %s' % learnedPolicy)
print()

# check whether goal can be reached from all start states
def testPolicy(maxSteps=20):
    for state in mc.allStates:
        (_, reward, _) = mc.generateEpisode(state, learnedPolicy, maxSteps, False).pop() # final step
        if reward == 1000:
            print(state.clingoString() + '\t\t\t✅')
        else:
            print(state.clingoString() + '\t\t\t❌')
        
testPolicy()
