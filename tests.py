from entities import *

# test whether goal can be reached from all start states
def testPolicy(policy, mc, maxEpisodeLength):
	finalState = State({PartState('on(a,table)'),PartState('on(b,a)'), PartState('on(c,table)')})#, PartState('on(d,table)')})
	finalAction = Action('move(c,b)')
	num_steps = []

	for state in mc.allStates:
		steps = mc.generateEpisode(state, policy, maxEpisodeLength, False, True)
		if steps:
			num_steps.append(len(steps))
			(s, _, a) = steps.pop() # final step
			if s == finalState and a == finalAction:
				print(f'{str(state):<80} {"✅":>1}')
			else:
				print(f'{str(state):<80} {"❌":>1}')
		else:
			print(f'{str(state):<80} {"✅":>1}') # empty episode means start == goal

	print('Avg steps: ' + str(sum(num_steps)/len(num_steps)))
	