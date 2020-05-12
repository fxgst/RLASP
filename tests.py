from entities import *

# test whether goal can be reached from all start states
def testPolicy(policy, mc, maxSteps=10):
	finalState = State({PartState('on(a,table)'),PartState('on(b,a)'), PartState('on(c,table)')})
	finalAction = Action('move(c,b)')
	num_steps = []

	for state in mc.allStates:
		steps = mc.generateEpisode(state, policy, maxSteps, False, True)
		if steps:
			num_steps.append(len(steps))
			(state, _, action) = steps.pop() # final step
			if state == finalState and action == finalAction:
				print(f'{str(state):<80} {"✅":>1}')
			else:
				print(f'{str(state):<80} {"❌":>1}')
		else:
			print(f'{str(state):<80} {"✅":>1}') # empty episode means start == goal

	print('Avg steps: ' + str(sum(num_steps)/len(num_steps)))
	