from entities import *
import matplotlib
import matplotlib.pyplot as plt
import statistics
import pickle
import numpy as np
from BlocksWorld import *
from MonteCarlo import *

number_runs 		= 5
number_episodes 	= 50
max_episode_len		= 8
mode 				= 'mean'
test_name			= '4b_50e_5r'
plot_every_n		= 1#number_episodes/100

def plot(arr, every_n):
	matplotlib.use('pgf')
	matplotlib.rcParams.update({
		'pgf.texsystem': 'pdflatex',
		'font.family': 'serif',
		'text.usetex': True,
		'pgf.rcfonts': False,
	})
	plt.plot(range(0, len(arr), every_n), arr[::every_n])
	plt.ylabel('return ratio')
	plt.xlabel('number of episodes')
	plt.savefig('./testdata/' + test_name + '_' + mode + '.pgf')
	plt.savefig('./testdata/' + test_name + '_' + mode + '.png', dpi=400)

def test(): 
	a = np.empty(number_runs, dtype=object)
	for i in range(0, number_runs):
		with open('./testdata/' + test_name + '_' + mode + f'_{i}.pkl', 'rb') as f:
			a[i] = pickle.load(f)
	
	if mode == 'median':
		result = [statistics.median(k) for k in zip(*a)]
	else: 
		result = [statistics.mean(k) for k in zip(*a)]

	plot(result, plot_every_n)

def generateRuns():
	blocksWorld = BlocksWorld()
	print('Generating runs...')

	for i in range(0, number_runs):
		print(i)
		mc = MonteCarlo(blocksWorld)
		mc.learnPolicy(maxEpisodeLength=max_episode_len, gamma=1, numberEpisodes=number_episodes)

		with open('./testdata/' + test_name + '_' + mode + f'_{i}.pkl', 'wb') as f:
			pickle.dump(mc.returnRatios, f)

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
	