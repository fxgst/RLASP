import clingo


class ClingoBridge:
    def __init__(self):
        self.output = []
        self.ctl = clingo.Control()  # Control object for the grounding/solving process

    def on_model(self, m: clingo.Model):
        self.output.append(m.symbols(False, False, True, False, False))

    def add_file(self, path: str):
        self.ctl.load(path)

    def run(self, programs: list, n: int = 0):
        self.ctl.configuration.solve.models = n  # create all stable models
        files = []

        # add programs to list of files for ASP program
        for program in programs:
            self.ctl.add(program[0], [], program[1])
            files.append((program[0], []))

        # ground & solve ASP program
        self.ctl.ground(files)
        self.ctl.solve(on_model=self.on_model)
