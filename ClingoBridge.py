import clingo


class ClingoBridge:
    def __init__(self):
        """Create a bridge to the ASP program.
        """
        self.output = []
        self.ctl = clingo.Control()  # Control object for the grounding/solving process

    def on_model(self, m: clingo.Model):
        """Appends an answer set to the output.

        :param m: a clingo model
        """
        self.output.append(m.symbols(False, False, True, False, False))

    def add_file(self, path: str):
        """Add a file in path to the control.

        :param path: the file path
        """
        self.ctl.load(path)

    def run(self, programs: list, n: int = 0):
        """Run a list of ASP programs.

        :param programs: a list of atoms or ASP programs
        :param n: number of answer sets to generate, 0 calculates all answer sets
        """
        self.ctl.configuration.solve.models = n
        files = []

        # add programs to list of files for ASP program
        for program in programs:
            self.ctl.add(program[0], [], program[1])
            files.append((program[0], []))

        # ground & solve ASP program
        self.ctl.ground(files)
        self.ctl.solve(on_model=self.on_model)
