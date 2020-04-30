from BlocksWorld import *
from entities import State
from random import randint
from collections import deque

class MonteCarlo:
    def __init__(self, blocksWorld: BlocksWorld):
        self.blocksWorld = blocksWorld
        self.allStates = self.blocksWorld.generateAllStates()
        
    def generateRandomPolicy(self):
        policy = dict()
        for i in range(0, len(self.allStates)):
            state = self.allStates[i]
            policy[state] = self.getRandomAction(state)

        return policy

    def getRandomAction(self, state: State):
        availableActions = self.blocksWorld.getAvailableActions(state)
        # randomly choose one applicable action
        rnd = randint(0, len(availableActions)-1)
        actions, reward = availableActions[rnd] 
        actions = actions if actions != [] else None # goal state has no applicable move

        return actions, reward

    # TODO: heuristic for resonable number of maxSteps in episode
    def generateEpisode(self, state: State, policy: dict, maxSteps, exploringStarts, onPolicy=True) -> deque:
        episode = deque() # deque allows much faster appending than array
        if exploringStarts and onPolicy:
            policy[state] = self.getRandomAction(state) # slow (clingo IO)

        count = 0
        while True:
            if count >= maxSteps:
                #print('Max steps exceeded')
                episode.append((state, -100, None))
                break

            if onPolicy:
                actions, reward = policy.get(state)
            else:
                actions, reward = self.getRandomAction(state) # slow (clingo IO)

            if actions == None:
                #print('Goal reached!')
                episode.append((state, reward, None))
                break

            action = actions[0]
            episode.append((state, reward, action))
            # perform one action
            state = self.blocksWorld.performMove(action, state) # slow (clingo IO)
            count += 1

        return episode

    def learnPolicy(self, maxEpisodeLength, gamma, episodes):
        """ First visit exploring starts Monte Carlo evaluation of policy P """

        print('Initializing...')
        Q = dict()                      # {state {action : average value}}
        Returns = dict()                # {state {action : number of experiences}}
        P = self.generateRandomPolicy() # {state : (action, reward)}

        print('Learning...')
        for _ in range(0, episodes):
            rnd = randint(0, len(self.allStates)-1)
            startState = self.allStates[rnd]
            episode = self.generateEpisode(startState, P, maxEpisodeLength, exploringStarts=True) # very slow

            g_return = 0
            for t in range(len(episode)-1-1, 0-1, -1):
                g_return = gamma * g_return + episode[t+1][1]

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
                
                    # greedily choose best action with max value
                    best_action = action_t
                    argMax = -10000
                    for action, value in Q[state_t].items():
                        if value > argMax:
                            argMax = value
                            best_action = action

                    # update policy with best action
                    tmp = list(P[state_t])
                    tmp[0] = [best_action]
                    P[state_t] = tuple(tmp)
        
        print('Done!')
        return P
