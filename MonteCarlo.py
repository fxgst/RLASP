from BlocksWorld import *
from entities import State
from random import randint, random
from collections import deque


class MonteCarlo:
    def __init__(self, blocks_world: BlocksWorld):
        self.blocksWorld = blocks_world
        self.returnRatios = []
        self.Q = dict()  # {state : {action : value}}

    # max_episode_length should be at least 2*(n-1)
    def generate_episode(self, state: State, policy: dict, max_episode_length: int, planning_factor: float,
                         plan_on_empty_policy: bool, planning_horizon: int, exploring_starts: bool, on_policy: bool) -> deque:
        episode = deque()  # deque allows much faster appending than array
        actions = self.get_initial_actions(state)  # clingo IO

        count = 0
        while count <= max_episode_length:
            if planning_factor <= random():
                if (state in policy) and on_policy:
                    if exploring_starts and count == 0:
                        action = self.get_random_action(actions)
                    else:
                        action = policy[state]
                elif plan_on_empty_policy:
                    action = self.plan_action(state, planning_horizon)
                else:
                    action = self.get_random_action(actions)
            else:
                action = self.plan_action(state, planning_horizon)
                if state in policy:
                    policy_action = policy[state]
                    q_value_policy_action = self.Q[state][policy_action]
                    q_value_planning_action = self.Q[state][action]

                    if q_value_planning_action < q_value_policy_action:
                        action = policy_action

            if action is None:
                # goal reached
                break

            (nextState, nextActions, _, nextReward, _) = self.blocksWorld.next_step(state, action, t=1)  # clingo IO
            episode.append((state, nextReward, action))
            state = nextState
            actions = nextActions
            count += 1

        return episode

    def learn_policy(self, max_episode_length: int, gamma: float, number_episodes: int, planning_factor: float,
                     plan_on_empty_policy: bool, planning_horizon: int) -> dict:
        """ First-visit Exploring Starts Monte Carlo evaluation of policy P """

        Visits = dict()  # {state : {action : number of experiences}}
        P = dict()  # {state : action}

        print('Learning...')
        for _ in range(0, number_episodes):
            start_state = self.blocksWorld.get_random_start_state()
            episode = self.generate_episode(start_state, P, max_episode_length, planning_factor, plan_on_empty_policy,
                                            planning_horizon, exploring_starts=True, on_policy=True)  # clingo IO

            g_return = 0
            for t in range(len(episode) - 1, 0 - 1, -1):
                g_return = gamma * g_return + episode[t][1]

                state_t = episode[t][0]
                action_t = episode[t][2]

                # return ratio for benchmarking
                if t == 0:
                    return_ratio = self.calculate_return_ratio(start_state, g_return, -(max_episode_length + 1))  # clingo IO
                    self.returnRatios.append(return_ratio)

                is_first_visit = True
                for before_t in range(0, t):
                    if episode[before_t][0] == state_t and episode[before_t][2] == action_t:
                        is_first_visit = False
                        break

                if is_first_visit:
                    if state_t not in self.Q:
                        self.Q[state_t] = dict()
                        Visits[state_t] = dict()
                        # initialize all possible actions in state_t with zero
                        for a in self.get_initial_actions(state_t):  # clingo IO
                            Visits[state_t][a] = 0
                            self.Q[state_t][a] = 0

                    # predict/evaluate: calculate average value for state-action pair
                    Visits[state_t][action_t] += 1
                    self.Q[state_t][action_t] += (g_return - self.Q[state_t][action_t]) / Visits[state_t][action_t]

                    # control/improve: use greedy exploration: choose action with highest return
                    P[state_t] = self.greedy_action(self.Q[state_t].items(), action_t)

        print('Done!')
        return P

    # auxiliary methods

    def greedy_action(self, actions, action_t):
        best_action = action_t
        arg_max = -100000
        for action, value in actions:
            if value == arg_max and randint(0, 1) == 1:
                best_action = action
            elif value > arg_max:
                arg_max = value
                best_action = action

        return best_action

    def get_initial_actions(self, state: State) -> list:
        (_, availableActions, _, _, _) = self.blocksWorld.next_step(state, None, t=0)  # clingo IO
        return availableActions

    def get_random_action(self, actions: list):
        if len(actions) > 0:
            rnd = randint(0, len(actions) - 1)
            return actions[rnd]
        return None

    def plan_action(self, state, planning_horizon):
        # plan and choose action according to planning component
        (_, _, bestAction, _, _) = self.blocksWorld.next_step(state, None, t=planning_horizon)  # clingo IO
        return bestAction

    def calculate_return_ratio(self, start_state: State, episode_reward: float, minimal_reward: float) -> float:
        (_, _, _, _, maxReward) = self.blocksWorld.next_step(start_state, None,
                                                             2 * (len(start_state.locations) - 1))  # clingo IO
        return (episode_reward - minimal_reward) / (maxReward - minimal_reward)
