from entities import *
import matplotlib
import matplotlib.pyplot as plt
import statistics
import pickle
import numpy as np
from BlocksWorld import *
from MonteCarlo import *
from testparms import *

test_name = str(num_blocks) + 'b_' + str(number_episodes) + 'e' + '_' + str(number_runs) + 'r'+ '_' + str(planning_factor) + 'pf' + ('_poep' if  plan_on_empty_policy else '') + (f'_{planning_horizon}phz' if (planning_factor != 0 or plan_on_empty_policy) else '')
plot_every_n = int(number_episodes/plot_points)

def plot(plotname, arrays):
	matplotlib.use('pgf')
	matplotlib.rcParams.update({
		'pgf.texsystem': 'pdflatex',
		'font.family': 'serif',
		'text.usetex': True,
		'pgf.rcfonts': False,
	})
	for arr in arrays:
		plt.plot(range(0, len(arr[0]), plot_every_n), arr[0][::plot_every_n], '.', label=arr[1])

	plt.legend(bbox_to_anchor=(0, 0.93, 1, 0.2), loc="upper left", mode='expand', ncol=4)
	plt.ylim(0, 1)
	plt.ylabel('return ratio')
	plt.xlabel('number of episodes')
	plt.savefig('./testdata/' + plotname + '.pgf')
	plt.savefig('./testdata/' + plotname + '.png', dpi=400)

def plotMultiple(plotname, filenames):
	toPlot = []
	for filename in filenames:
		toPlot.append((loadPlotData(filename[0]), filename[1]))
	plot(plotname, toPlot)

def loadPlotData(filename): 
	a = np.empty(number_runs, dtype=object)
	for i in range(0, number_runs):
		with open('./testdata/' + filename + f'_{i}.pkl', 'rb') as f:
			a[i] = pickle.load(f)
	
	if mode == 'median':
		result = [statistics.median(k) for k in zip(*a)]
	else:
		result = [statistics.mean(k) for k in zip(*a)]

	return result

def cacheBlocksWorld(blocksWorld, numberOfBlocks):
	if numberOfBlocks < 10:
		with open('./testdata/' + str(numberOfBlocks) + '_blocksworld.pkl', 'wb') as f:
			pickle.dump(blocksWorld.allStates, f)

def generateRuns(pathToBlocksWorld = None):
	if pathToBlocksWorld:
		print('Loading blocks world...')
		blocksWorld = BlocksWorld(pathToBlocksWorld)
	else:
		print('Generating blocks world...')
		blocksWorld = BlocksWorld()
		cacheBlocksWorld(blocksWorld, num_blocks)
	print('Done!')

	print('Generating runs...')
	for i in range(0, number_runs):
		print('Run ' + str(i))
		mc = MonteCarlo(blocksWorld)
		mc.learnPolicy(maxEpisodeLength=max_episode_len, gamma=1, numberEpisodes=number_episodes, planningFactor=planning_factor, planOnEmptyPolicy=plan_on_empty_policy, planningHorizon=planning_horizon)
		with open('./testdata/' + test_name + f'_{i}.pkl', 'wb') as f:
			pickle.dump(mc.returnRatios, f)

# test whether goal can be reached from all start states
def testPolicy(policy, mc, maxEpisodeLength):
	#finalState = State({PartState('on(a,table)'),PartState('on(b,a)'), PartState('on(c,b)'), PartState('on(d,c)'), PartState('on(e,table)')})
	finalState = State({PartState('on(a,table)'), PartState('on(b,a)'), PartState('on(c,b)'), PartState('on(d,c)'), PartState('on(e,d)'), PartState('on(f,e)'), PartState('on(g,f)'), PartState('on(h,g)'), PartState('on(i,h)'), PartState('on(j,table)')})
	finalAction = Action('move(j,i)')#Action('move(e,d)')
	num_steps = []

	for state in mc.blocksWorld.allStates:
		steps = mc.generateEpisode(state, policy, maxEpisodeLength, 0, False, 0, False, True)
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
	