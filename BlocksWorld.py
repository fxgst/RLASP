from ClingoBridge import *
from random import randint
from entities import *
import numpy as np

class BlocksWorld:
    def __init__(self, blocks: set, goal: State):
        self.blocks = blocks
        self.clingo = ClingoBridge()
        self.goal = goal

    def performMove(self, move: Move, state: State) -> State:
        self.clingo = ClingoBridge() # reset clingo
        program = []
        # add blocks to program
        program.append(('base', ''.join([block.clingoString() for block in self.blocks])))
        # add rule of form { on(c, b, 0); on(a, table, 0); on(b, a, 0) } = 2.
        rule = '{ ' + (''.join([on.clingoString() for on in state.locations])).replace('.', ';').rsplit(';', 1)[0] + ' } = ' + str(len(self.blocks)-1) + '.'
        program.append(('base', rule))
        # add move to program
        program.append(('base', move.clingoString()))
        # add main program file
        self.clingo.addFile('updateState.lp')

        self.clingo.run(program)

        return self.parseState(self.clingo.output[0], 0)

    def generateAllStates(self):
        self.clingo = ClingoBridge() # reset clingo
        # add blocks to file
        blocks = (('base', ''.join([block.clingoString() for block in self.blocks])))
        # add main program file
        self.clingo.addFile('enumerateStates.lp')

        self.clingo.run([blocks])
        output = self.clingo.output

        states = np.full((len(output)), object)
        for i in range(0, len(output)):
            states[i] = self.parseState(output[i], 0)

        return states

    def getRandomStartState(self) -> State:
        self.clingo = ClingoBridge() # reset clingo
        allStates = self.generateAllStates()
        # choose random start state
        rnd = randint(0, len(allStates)-1)
        return allStates[rnd]

    def getAvailableActions(self, state: State, planAhead = 0) -> list:
        self.clingo = ClingoBridge() # reset clingo

        facts = []
        # add state to ASP program
        facts.append(('base', ''.join([on.clingoString() for on in state.locations])))
        # add blocks to ASP program
        facts.append(('base', ''.join([block.clingoString() for block in self.blocks])))
        # add goal state to ASP program
        facts.append(('base', ''.join([subgoal.clingoString() for subgoal in self.goal.locations])))
        # declare constant in ASP program
        facts.append(('base', '#const t = %i.' % planAhead))
        # add main program file
        self.clingo.addFile('nextMoves.lp')

        self.clingo.run(facts)
        available_actions = []

        for answerSet in self.clingo.output:
            moves = []
            reward = None

            for atom in answerSet:
                if (atom.name == 'move'):
                    topBlock = Block(atom.arguments[0])
                    bottomBlock = Block(atom.arguments[1])
                    moves.append(Move(topBlock, bottomBlock, 0))
                elif (atom.name == 'totalReward'):
                    reward = atom.arguments[0].number
                else:
                    print('ERROR: unexpected atom')

            available_actions.append((moves, reward))

        return available_actions
        
    def parseState(self, raw_state, time) -> State:
        state = [] # state defined by one on(X,Y) per block
        for on in raw_state:
            topBlock = Block(on.arguments[0])
            bottomBlock = Block(on.arguments[1])
            state.append(On(topBlock, bottomBlock, time))
        return State(set(state))
