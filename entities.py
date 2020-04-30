class Block:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '%s' % self.name
    
    def clingoString(self):
        return 'block(%s). ' % self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)
      
class SubGoal:
    def __init__(self, x: Block, y: Block):
        self.x = x
        self.y = y

    def __repr__(self):
        return '%s is on top of %s' % (self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def clingoString(self):
        return 'subgoal(%s, %s). ' % (self.x, self.y)

    def __hash__(self):
        return hash((self.x, self.y))

class On:
    def __init__(self, x: Block, y: Block, time = 0):
        self.x = x
        self.y = y
        self.time = time

    def __repr__(self):
        return '%s is on top of %s' % (self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def clingoString(self):
        return 'on(%s, %s, %s). ' % (self.x, self.y, self.time)

    def __hash__(self):
        return hash((self.x, self.y))

class Move:
    def __init__(self, x: Block, y: Block, time: int):
        self.x = x
        self.y = y
        self.time = time

    def __repr__(self):
        return '%s: Move %s on top of %s' % (self.time, self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.time == other.time

    def clingoString(self):
        return 'move(%s, %s, %s). ' % (self.x, self.y, self.time)

    def __hash__(self):
        return hash((self.x, self.y, self.time))

class State:
    def __init__(self, locations: set):
        self.locations = locations

    def clingoString(self):
        return ''.join([location.clingoString() for location in self.locations])

    def __repr__(self):
        return  ', '.join([location.__repr__() for location in self.locations])

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return set([str(a) for a in self.locations]) == set([str(a) for a in other.locations])        

    def __hash__(self):
        return hash(frozenset([str(a) for a in self.locations]))
