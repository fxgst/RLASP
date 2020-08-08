from BlocksWorld import *
from entities import State
from random import randint, random
from collections import deque


class MonteCarlo:
    def __init__(self, blocks_world: BlocksWorld, max_episode_length: int, planning_factor: float, plan_on_empty_policy: bool, planning_horizon: int, exploring_starts: bool = True, on_policy: bool = True):
        """Sets all required properties for the learning process.

        :param blocks_world: the blocks world
        :param max_episode_length: the maximum number of steps in an episode, should be at least 2*(n-1), n = number of blocks
        :param planning_factor: the probability for invoking the planning component at each step
        :param plan_on_empty_policy: enable/disable the use of planning on an empty policy entry
        :param planning_horizon: how far to plan ahead
        :param exploring_starts: enable/disable exploring starts
        :param on_policy: enable/disable on-policy behavior
        """
        self.blocks_world = blocks_world
        self.max_episode_length = max_episode_length
        self.planning_factor = planning_factor
        self.plan_on_empty_policy = plan_on_empty_policy
        self.planning_horizon = planning_horizon
        self.exploring_starts = exploring_starts
        self.on_policy = on_policy
        self.return_ratios = []
        self.Q = dict()  # {state : {action : value}}

    def generate_episode(self, state: State, policy: dict) -> deque:
        """Generates a single episode from a start state onwards.

        :param state: the initial state
        :param policy: the current policy
        :return: a sequence of state, reward, action
        """
        episode = deque()  # deque allows faster appending than array
        actions = self.get_initial_actions(state)  # clingo IO

        count = 0
        while count <= self.max_episode_length:
            if self.planning_factor <= random():
                if (state in policy) and self.on_policy:
                    if self.exploring_starts and count == 0:
                        action = self.get_random_action(actions)
                    else:
                        action = policy[state]
                elif self.plan_on_empty_policy:
                    action = self.plan_action(state, self.planning_horizon)
                else:
                    action = self.get_random_action(actions)
            else:
                action = self.plan_action(state, self.planning_horizon)
                if state in policy:
                    policy_action = policy[state]
                    q_value_policy_action = self.Q[state][policy_action]
                    q_value_planning_action = self.Q[state][action]

                    if q_value_planning_action < q_value_policy_action:
                        action = policy_action

            if action is None:
                # goal reached
                break

            (nextState, nextActions, _, nextReward, _) = self.blocks_world.next_step(state, action, t=1)  # clingo IO
            episode.append((state, nextReward, action))
            state = nextState
            actions = nextActions
            count += 1

        return episode

    def learn_policy(self, gamma: float, number_episodes: int) -> dict:
        """Uses a first-visit Exploring Starts Monte Carlo evaluation method to evaluate policy P.

        :param gamma: the discounting factor (use only when no planning is used; set to 1 if planning is used)
        :param number_episodes: the number of episodes to run
        :return: the learned policy as a state-action mapping
        """

        Visits = dict()  # {state : {action : number of experiences}}
        P = dict()  # {state : action}

        print('Learning...')
        for _ in range(0, number_episodes):
            start_state = self.blocks_world.get_random_start_state()
            episode = self.generate_episode(start_state, P)  # clingo IO

            g_return = 0
            for t in range(len(episode) - 1, 0 - 1, -1):
                g_return = gamma * g_return + episode[t][1]

                state_t = episode[t][0]
                action_t = episode[t][2]

                # return ratio for benchmarking
                if t == 0:
                    return_ratio = self.calculate_return_ratio(start_state, g_return, -(self.max_episode_length + 1))  # clingo IO
                    self.return_ratios.append(return_ratio)

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

    def greedy_action(self, actions: dict, action_t: Action) -> Action:
        """Chooses the action with the highest value.

        :param actions: action-value pairs
        :param action_t: the action that happened at time t
        :return: the action with highest value, ties broken arbitrarily
        """
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
        """Retrieves the applicable actions for a given state.

        :param state: the current state
        :return: a list of applicable actions in that state
        """
        (_, availableActions, _, _, _) = self.blocks_world.next_step(state, None, t=0)  # clingo IO
        return availableActions

    def get_random_action(self, actions: list) -> Action:
        """Picks a random action from a list of actions.

        :param actions: a list of possible actions
        :return: a random action
        """
        if len(actions) > 0:
            rnd = randint(0, len(actions) - 1)
            return actions[rnd]
        return None

    def plan_action(self, state: State, planning_horizon: int) -> Action:
        """Getting an action recommended by the planning component.

        :param state: the current state
        :param planning_horizon: how many steps to plan ahead
        :return: an action recommended by the planning component
        """
        (_, _, bestAction, _, _) = self.blocks_world.next_step(state, None, t=planning_horizon)  # clingo IO
        return bestAction

    def calculate_return_ratio(self, start_state: State, episode_reward: float, minimal_reward: float) -> float:
        """Calculates the return ratio of an episode.

        :param start_state: the initial state
        :param episode_reward: the actual reward achieved during that episode
        :param minimal_reward: the minimal reward the agent can achieve during an episode
        :return: the return ratio
        """
        (_, _, _, _, maxReward) = self.blocks_world.next_step(start_state, None, 2 * (len(start_state.locations) - 1))  # clingo IO
        return (episode_reward - minimal_reward) / (maxReward - minimal_reward)
