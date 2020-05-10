from ClingoBridge import *
from random import randint
from entities import *
import numpy as np

class BlocksWorld:
    # TODO: single knowledge base to not reset clingo each time
    
    def __init__(self, blocks: list, goal: State):
        self.blocks = blocks
        self.clingo = ClingoBridge()
        self.goal = goal

    def generateAllStates(self):
        self.clingo = ClingoBridge() # reset clingo

        blocks = ('base', ''.join([f'block({block}). ' for block in self.blocks]))
        self.clingo.addFile('initial-states.lp')
        self.clingo.run([blocks])
        output = self.clingo.output

        states = np.full((len(output)), object)
        for i in range(0, len(output)):
            states[i] = self.parseState(output[i])

        return states

    def getRandomStartState(self) -> State:
        self.clingo = ClingoBridge() # reset clingo
        allStates = self.generateAllStates()

        # choose random start state
        rnd = randint(0, len(allStates)-1)
        return allStates[rnd]

    def nextStep(self, state: State, action: Action, t=1):
        self.clingo = ClingoBridge() # reset clingo
        facts = []

        # add dynmaic rules
        facts.append(('base', ''.join([part_state.previousString() for part_state in state.locations])))
        facts.append(('base', ''.join([part_state.goalString() for part_state in self.goal.locations])))
        facts.append(('base', f'#const t = {t}.'))
        if action:
            facts.append(('base', action.clingoString()))

        # add static main program file
        self.clingo.addFile('blocksworld-mdp.lp')
        self.clingo.run(facts)
        output = self.clingo.output

        availableActions = []
        partStates = []
        maxReward = None
        nextReward = None
        bestAction = None
        goalReached = False

        answerSet = output[0] # there should only be one answer set
        for atom in answerSet:
            if (atom.name == 'executable'):
                availableActions.append(self.parseAction(atom))
            elif (atom.name == 'partState'):
                partStates.append(self.parsePartState(atom))
            elif (atom.name == 'bestAction'):
                bestAction = self.parseAction(atom)
            elif (atom.name == 'nextReward'):
                nextReward = atom.arguments[0].number
            elif (atom.name == 'maxReward'):
                maxReward = atom.arguments[0].number
            elif (atom.name == 'goal'):
                goalReached = True
            else:
                print('ERROR: unexpected atom')

        return (State(set(partStates)), availableActions, bestAction, nextReward, maxReward, goalReached)

    def parsePartState(self, atom) -> PartState:
        onPredicate = atom.arguments[0]
        topBlock = onPredicate.arguments[0]
        bottomBlock = onPredicate.arguments[1]
        return PartState(f'{topBlock},{bottomBlock}')

    def parseAction(self, atom) -> Action:
        movePredicate = atom.arguments[0]
        topBlock = movePredicate.arguments[0]
        bottomBlock = movePredicate.arguments[1]
        return Action(f'move({topBlock},{bottomBlock})')

    def parseState(self, atoms) -> State:
        partStates = []
        for partState in atoms:
            partStates.append(self.parsePartState(partState))
        return State(set(partStates))