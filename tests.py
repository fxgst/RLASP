from MonteCarlo import *


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
