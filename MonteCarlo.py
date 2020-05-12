from BlocksWorld import *
from entities import State
from random import randint
from collections import deque

class MonteCarlo:
    def __init__(self, blocksWorld: BlocksWorld):
        self.blocksWorld = blocksWorld
        self.allStates = self.blocksWorld.generateAllStates()

    def getRandomAction(self, state: State) -> Action:
        (_, availableActions, _, _, _) = self.blocksWorld.nextStep(state, None, t=0)
        # randomly choose one applicable action
        if len(availableActions) > 0:
            rnd = randint(0, len(availableActions)-1)
            return availableActions[rnd]

        return None 

    # TODO: heuristic for resonable number of maxSteps in episode
    def generateEpisode(self, state: State, policy: dict, maxSteps: int, exploringStarts: bool, onPolicy: bool) -> deque:
        episode = deque() # deque allows much faster appending than array
        if exploringStarts and onPolicy:
            policy[state] = self.getRandomAction(state) # slow (clingo IO)

        count = 0
        while True:
            if count >= maxSteps:
                break

            if onPolicy and (state in policy):
                action = policy[state]
            else:
                action = self.getRandomAction(state) # slow (clingo IO)
                policy[state] = action

            if action == None:
                # goal reached
                break

            (newState, _, _, nextReward, _) = self.blocksWorld.nextStep(state, action, t=1) # slow (clingo IO)
            episode.append((state, nextReward, action))
            state = newState
            count += 1

        return episode

    def learnPolicy(self, maxEpisodeLength: int, gamma: float, numberEpisodes: int) -> dict:
        """ First visit exploring starts Monte Carlo evaluation of policy P """

        print('Initializing...')
        Q = dict()        # {state {action : average value}}
        Returns = dict()  # {state {action : number of experiences}}
        P = dict()        # {state : action}

        print('Learning...')
        for _ in range(0, numberEpisodes):
            rnd = randint(0, len(self.allStates)-1)
            startState = self.allStates[rnd]
            episode = self.generateEpisode(startState, P, maxEpisodeLength, exploringStarts=True, onPolicy=True) # very slow

            g_return = 0
            for t in range(len(episode)-1, 0-1, -1):
                g_return = gamma * g_return + episode[t][1]

                state_t = episode[t][0]
                action_t = episode[t][2]

                firstVisit = True
                for before_t in range(0, t):
                    if episode[before_t][0] == state_t and episode[before_t][2] == action_t:
                        firstVisit = False
                        break

                if firstVisit:
                    if state_t not in Q:
                        Q[state_t] = dict()
                        Returns[state_t] = dict()
                    if action_t not in Q[state_t]:
                        Q[state_t][action_t] = 0
                        Returns[state_t][action_t] = 0

                    # calculate average value for state-action pair
                    n = Returns[state_t][action_t]
                    Q[state_t][action_t] = Q[state_t][action_t] + (g_return - Q[state_t][action_t]) / (n+1)
                    Returns[state_t][action_t] += 1
                
                    # use greedy exploration: choose action with highest return
                    best_action = action_t
                    argMax = -10000
                    for action, value in Q[state_t].items():
                        if value > argMax:
                            argMax = value
                            best_action = action

                    # update policy with best action
                    P[state_t] = best_action
        
        print('Done!')
        return P
