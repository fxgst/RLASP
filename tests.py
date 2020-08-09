import matplotlib
import matplotlib.pyplot as plt
import statistics
from MonteCarlo import *
from testparms import *

test_name = str(num_blocks) + 'b_' + str(number_episodes) + 'e' + '_' + str(number_runs) + 'r' + '_' + str(planning_factor) + 'pf' + ('_poep' if plan_on_empty_policy else '') + (f'_{planning_horizon}phz' if (planning_factor != 0 or plan_on_empty_policy) else '')
plot_every_n = int(number_episodes / plot_points)
test_data_path = './testdata/'


def plot(plot_name, arrays):
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
    plt.savefig(test_data_path + plot_name + '.pgf')
    plt.savefig(test_data_path + plot_name + '.png', dpi=400)


def plot_multiple(plot_name, filenames):
    to_plot = []
    for filename in filenames:
        to_plot.append((load_plot_data(filename[0]), filename[1]))
    plot(plot_name, to_plot)


def load_plot_data(filename):
    a = np.empty(number_runs, dtype=object)
    for i in range(0, number_runs):
        try:
            with open(test_data_path + filename + f'_{i}.pkl', 'rb') as f:
                a[i] = pickle.load(f)
        except FileNotFoundError:
            print(filename + ' not found')
            pass

    if mode == 'median':
        result = [statistics.median(k) for k in zip(*a)]
    else:
        result = [statistics.mean(k) for k in zip(*a)]

    return result


def cache_blocks_world(blocks_world, number_of_blocks):
    if number_of_blocks < 11:
        with open(test_data_path + str(number_of_blocks) + '_blocksworld.pkl', 'wb') as f:
            pickle.dump(blocks_world.allStates, f)


def generate_runs(path_to_blocks_world=None):
    if path_to_blocks_world:
        print('Loading blocks world...')
        blocks_world = BlocksWorld(path_to_blocks_world)
    else:
        print('Generating blocks world...')
        blocks_world = BlocksWorld()
        cache_blocks_world(blocks_world, num_blocks)
    print('Done!')

    print('Generating runs...')
    for i in range(0, number_runs):
        print('Run ' + str(i))
        mc = MonteCarlo(blocks_world, max_episode_length, planning_factor, plan_on_empty_policy, planning_horizon)
        mc.learn_policy(1, number_episodes)
        with open(test_data_path + test_name + f'_{i}.pkl', 'wb') as f:
            pickle.dump(mc.return_ratios, f)


def test_policy(policy, blocks_world, max_episode_length):
    """Test whether the goal state can be reached from each starting state for a given policy.
    For each state, a check will be printed if the goal state was reachable from that state, a cross otherwise.
    Note that this test is tailored for a blocks world of size 4, if other blocks worlds should be tested,
    the final_state and final_action fields have to be updated accordingly.

    :param policy: the policy to be evaluated
    :param blocks_world: a blocks world
    :param max_episode_length: the maximum number of steps before the test for that particular start state gets aborted
    """
    final_state = State({PartState('on(a,table)'), PartState('on(b,a)'), PartState('on(c,b)'), PartState('on(d,table)')})
    # final_state = State({PartState('on(a,table)'), PartState('on(b,a)'), PartState('on(c,b)'), PartState('on(d,c)'), PartState('on(e,d)'), PartState('on(f,e)'), PartState('on(g,f)'), PartState('on(h,g)'), PartState('on(i,h)'), PartState('on(j,table)')})
    final_action = Action('move(d,c)')
    num_steps = []
    mc = MonteCarlo(blocks_world, max_episode_length, 0, False, 0, False, True)

    for state in blocks_world.allStates:
        steps = mc.generate_episode(state, policy)
        if steps:
            num_steps.append(len(steps))
            (s, _, a) = steps.pop()  # final step
            if s == final_state and a == final_action:
                print(f'{str(state):<80} {"✅":>1}')
            else:
                print(f'{str(state):<80} {"❌":>1}')
        else:
            print(f'{str(state):<80} {"✅":>1}')  # empty episode means start == goal

    print('Average steps: ' + str(sum(num_steps) / len(num_steps)))
