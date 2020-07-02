from ClingoBridge import *
from random import randint
from entities import *
import numpy as np
import pickle

class BlocksWorld:
    def __init__(self, path = None):
        self.clingo = ClingoBridge()
        if path:
            with open(path, 'rb') as f:
                self.allStates = pickle.load(f)
        else:
            self.allStates = self.generateAllStates()

    def generateAllStates(self):
        self.clingo = ClingoBridge() # reset clingo

        base = ('base', '')
        self.clingo.addFile('initial-states.lp')
        self.clingo.run([base])
        output = self.clingo.output

        states = np.full((len(output)), object)
        for i in range(0, len(output)):
            states[i] = self.parseState(output[i])

        return states

    def nextStep(self, state: State, action: Action, t):
        self.clingo = ClingoBridge() # reset clingo
        facts = []

        # add dynmaic rules
        facts.append(('base', ''.join([part_state.clingoString() for part_state in state.locations])))
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

        answerSet = output[-1] # take last, most optimal output
        for atom in answerSet:
            if (atom.name == 'executable'):
                availableActions.append(self.parseAction(atom))
            elif (atom.name == 'state'):
                partStates.append(self.parsePartState(atom))
            elif (atom.name == 'bestAction'):
                bestAction = self.parseAction(atom)
            elif (atom.name == 'nextReward'):
                nextReward = atom.arguments[0].number
            elif (atom.name == 'maxReward'):
                maxReward = atom.arguments[0].number
            else:
                print(f'ERROR: unexpected atom "{atom.name}"')

        return (State(set(partStates)), availableActions, bestAction, nextReward, maxReward)

    def parsePartState(self, atom: clingo.Symbol) -> PartState:
        onPredicate = atom.arguments[0]
        topBlock = onPredicate.arguments[0]
        bottomBlock = onPredicate.arguments[1]
        return PartState(f'on({topBlock},{bottomBlock})')

    def parseAction(self, atom: clingo.Symbol) -> Action:
        movePredicate = atom.arguments[0]
        topBlock = movePredicate.arguments[0]
        bottomBlock = movePredicate.arguments[1]
        return Action(f'move({topBlock},{bottomBlock})')

    def parseState(self, atoms: list) -> State:
        partStates = []
        for partState in atoms:
            partStates.append(self.parsePartState(partState))
        return State(set(partStates))
        