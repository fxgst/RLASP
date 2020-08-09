class PartState:
    def __init__(self, id: str):
        self.id = id.replace(' ', '')

    def __repr__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def clingoString(self):
        return f'current({self.id}). '

    def __hash__(self):
        return hash(id)

class Action:
    def __init__(self, id: str):
        self.id = id.replace(' ', '')

    def __repr__(self):
        return self.id

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.id == other.id

    def clingoString(self):
        return f'action({self.id}). '

    def __hash__(self):
        return hash(self.id)

class State:
    def __init__(self, locations: set):
        self.locations = locations

    def __repr__(self):
        return  '{' + ', '.join([location.__repr__() for location in self.locations]) + '}'

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return set([str(a) for a in self.locations]) == set([str(a) for a in other.locations])   

    def clingoString(self):
        return ''.join([location.clingoString() for location in self.locations])

    def __hash__(self):
        return hash(frozenset([str(a) for a in self.locations]))
