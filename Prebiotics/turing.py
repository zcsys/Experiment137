class TuringMachine:
    def __init__(
        self,
        blank = '0',
        initial_state = 'q0',
        final_states = {'qf'}
        ):
        self.blank = blank
        self.transitions = {}
        self.final_states = final_states
        self.initialize(initial_state)

    def initialize(self, initial_state = 'q0'):
        self.head_position = 0
        self.state = initial_state

    def set_transition(
        self,
        state,
        symbol,
        next_state,
        write_symbol,
        direction
        ):
        self.transitions[(state, symbol)] = (
            next_state,
            write_symbol,
            direction
        )

    def step(self):
        current_symbol = (
            self.tape[self.head_position]
            if self.head_position < len(self.tape)
            else self.blank_symbol
        )
        key = (self.state, current_symbol)
        if key in self.transitions:
            next_state, write_symbol, direction = self.transitions[key]

            if self.head_position < len(self.tape):
                self.tape[self.head_position] = write_symbol
            else:
                self.tape.append(write_symbol)

            if direction == 'R':
                self.head_position += 1
            elif direction == 'L' and self.head_position > 0:
                self.head_position -= 1

            self.state = next_state
        else:
            print(
                f"No transition for state {self.state}"
                f"and symbol {current_symbol}. Halting."
            )
            return False
        return True

    def run(self, tape, show = False):
        self.tape = list(tape)
        while self.state not in self.final_states:
            if not self.step():
                break
        if show:
            print("Final tape:", ''.join(self.tape))


if __name__ == "__main__":
    # Solving XOR
    tm = TuringMachine()

    tm.set_transition('q0', '0', 'q1', tm.blank, 'R')
    tm.set_transition('q0', '1', 'q2', tm.blank, 'R')
    tm.set_transition('q1', '0', 'qf', '0', '')
    tm.set_transition('q1', '1', 'qf', '1', '')
    tm.set_transition('q2', '0', 'qf', '1', '')
    tm.set_transition('q2', '1', 'qf', '0', '')

    inputs = ['00', '01', '10', '11']

    for input in inputs:
        tm.run(input, True)
        tm.initialize()
