from BlocksWorld import *
from entities import State
from random import randint, random
from collections import deque

class MonteCarlo:
    def __init__(self, blocksWorld: BlocksWorld):
        self.blocksWorld = blocksWorld
        self.allStates = self.blocksWorld.generateAllStates() # clingo IO
        self.returnRatios = []

    # maxEpisodeLength should be at least 2*(n-1)
    def generateEpisode(self, state: State, policy: dict, maxEpisodeLength: int, planningFactor: float, planOnEmptyPolicy: bool, planningHorizon: int, exploringStarts: bool, onPolicy: bool) -> deque:
        episode = deque() # deque allows much faster appending than array
        actions = self.getInitialActions(state) # clingo IO

        count = 0
        while count <= maxEpisodeLength:
            if planningFactor <= random():
                if (state in policy) and onPolicy:
                    if exploringStarts and count == 0:
                        action = self.getRandomAction(actions)
                    else:
                        action = policy[state]
                elif planOnEmptyPolicy:
                    action = self.planAction(state, planningHorizon)
                else:
                    action = self.getRandomAction(actions)  
            else:
                action = self.planAction(state, planningHorizon)

            if action == None:
                # goal reached
                break

            (nextState, nextActions, _, nextReward, _) = self.blocksWorld.nextStep(state, action, t=1) # clingo IO
            episode.append((state, nextReward, action))
            state = nextState
            actions = nextActions
            count += 1

        return episode

    def learnPolicy(self, maxEpisodeLength: int, gamma: float, numberEpisodes: int, planningFactor: float, planOnEmptyPolicy: bool, planningHorizon: int) -> dict:
        """ First-visit Exploring Starts Monte Carlo evaluation of policy P """

        print('Initializing...')
        Q = dict()        # {state : {action : average value}}
        Visits = dict()   # {state : {action : number of experiences}}
        P = dict()        # {state : action}

        print('Learning...')
        for _ in range(0, numberEpisodes):
            rnd = randint(0, len(self.allStates)-1)
            startState = self.allStates[rnd]
            episode = self.generateEpisode(startState, P, maxEpisodeLength, planningFactor, planOnEmptyPolicy, planningHorizon, exploringStarts=True, onPolicy=True) # clingo IO

            g_return = 0
            for t in range(len(episode)-1, 0-1, -1):
                g_return = gamma * g_return + episode[t][1]

                state_t = episode[t][0]
                action_t = episode[t][2]

                # return ratio for benchmarking
                if t == 0:
                    rewardRatio = self.calculateReturnRatio(startState, g_return, -(maxEpisodeLength+1)) # clingo IO
                    self.returnRatios.append(rewardRatio)

                firstVisit = True
                for before_t in range(0, t):
                    if episode[before_t][0] == state_t and episode[before_t][2] == action_t:
                        firstVisit = False
                        break

                if firstVisit:
                    if state_t not in Q:
                        Q[state_t] = dict()
                        Visits[state_t] = dict()
                        # initialize all possible actions in state_t with zero
                        for a in self.getInitialActions(state_t):
                            Visits[state_t][a] = 0
                            Q[state_t][a] = 0

                    # predict/evaluate: calculate average value for state-action pair
                    Visits[state_t][action_t] += 1
                    Q[state_t][action_t] += (g_return - Q[state_t][action_t]) / Visits[state_t][action_t]
                    
                    # control/improve: use greedy exploration: choose action with highest return
                    P[state_t] = self.greedyAction(Q[state_t].items(), action_t)
        
        print('Done!')
        return P

    # auxiliary methods

    def greedyAction(self, actions, action_t):
        best_action = action_t
        argMax = -100000
        for action, value in actions:
            if value == argMax and randint(0,1) == 1:
                best_action = action
            elif value > argMax:
                argMax = value
                best_action = action

        return best_action

    def getInitialActions(self, state: State) -> list:
        (_, availableActions, _, _, _) = self.blocksWorld.nextStep(state, None, t=0) # clingo IO   
        return availableActions

    def getRandomAction(self, actions: list):
        if len(actions) > 0:
            rnd = randint(0, len(actions)-1)
            return actions[rnd]
        return None

    def planAction(self, state, planningHorizon):
        # plan and choose action according to planning component
        (_, _, bestAction, _, _) = self.blocksWorld.nextStep(state, None, t=planningHorizon) # clingo IO
        return bestAction

    def calculateReturnRatio(self, startState: State, episodeReward: float, minimalReward: float) -> float:
        (_, _, _, _, maxReward) = self.blocksWorld.nextStep(startState, None, len(startState.locations)*2) # clingo IO
        return (episodeReward - minimalReward) / (maxReward - minimalReward)
