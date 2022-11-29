import copy
import pickle
from typing import List, Optional, Union, Set


class Literal:
    def __init__(self, tokens: List[int]):
        self.tokens = tokens
        self.pointer = -1
        self.satisfy = False

    def advance(self, word_id: int):
        # token matches the next token in constraint
        if word_id == self.tokens[self.pointer + 1]:
            self.pointer += 1
        else:
            self.pointer = -1

        if self.pointer == len(self.tokens) - 1:
            self.satisfy = True


class Clause:
    def __init__(self, idx: int, phrases: List[List[int]], key_phrases: List[List[int]]):
        self.idx = idx
        self.literals = [Literal(p) for p in phrases]
        self.key_phrases = key_phrases
        self.satisfy = False

    def advance(self, word_id: int):
        for literal in self.literals:
            literal.advance(word_id)
            if literal.satisfy:
                self.satisfy = True

    def __str__(self):
        return f'clause(id={self.idx}, phrases={[l.tokens for l in self.literals]}, satisfy={self.satisfy})'


class ConstrainedHypothesis:

    def __init__(self,
                 constraint_list: List[List[List[int]]],
                 key_constraint_list: List[List[List[int]]],
                 eos_id: Union[int, list]
                 ) -> None:
        self.eos_id = eos_id if isinstance(eos_id, list) else [eos_id]
        self.orders = []

        self.clauses = []
        for idx, (clause, k_clause) in enumerate(zip(constraint_list, key_constraint_list)):
            self.clauses.append(Clause(idx=idx, phrases=clause, key_phrases=k_clause))

    def __len__(self) -> int:
        """
        :return: The number of constraints.
        """
        return len(self.clauses)

    def __str__(self) -> str:
        return '\n'.join([str(c) for c in self.clauses])

    def size(self) -> int:
        """
        :return: the number of constraints
        """
        return len(self.clauses)

    def num_met(self) -> int:
        """
        :return: the number of constraints that have been met.
        """
        return sum([int(c.satisfy) for c in self.clauses])

    def met_order(self) -> tuple:
        """
        :return: the ids of satisfied clauses.
        """
        return tuple(sorted(self.orders))

    def clause_in_process(self) -> tuple:
        """
        :return: the index of clause that's in generation.
        """
        in_process = []
        for clause in self.clauses:
            if clause.satisfy:
                continue

            if any([literal.pointer > -1 for literal in clause.literals]):
                in_process.append(clause.idx)
                assert all(literal.pointer < len(literal.tokens) - 1 and not literal.satisfy
                           for literal in clause.literals)

        return tuple(in_process)

    def met_process(self) -> tuple:
        return tuple(sorted(self.orders + list(self.clause_in_process())))

    def num_needed(self) -> int:
        """
        :return: the number of un-met constraints.
        """
        return self.size() - self.num_met()

    def finished(self) -> bool:
        """
        Return true if all the constraints have been met.

        :return: True if all the constraints are met.
        """
        return self.num_needed() == 0

    def is_valid(self, wordid: int) -> bool:
        """
        Ensures </s> is only generated when the hypothesis is completed.

        :param wordid: The wordid to validate.
        :return: True if all constraints are already met or the word ID is not the EOS id.
        """
        return self.finished() or wordid not in self.eos_id

    def eos(self) -> list:
        """
        :return: Return EOS id.
        """
        return self.eos_id

    def phrase_to_look_ahead(self) -> List[List[int]]:
        """

        :return: the literals in unsatisfied clauses
        """
        look_ahead_phrases = []
        for clause in self.clauses:
            if not clause.satisfy:
                look_ahead_phrases.extend(clause.key_phrases)

        return look_ahead_phrases

    def continue_to_look_ahead(self) -> List[List[int]]:
        """

        :return: the literals in process
        """
        look_ahead_continues = []

        for clause in self.clauses:
            if clause.satisfy:
                continue

            for literal in clause.literals:
                assert literal.pointer < len(literal.tokens) - 1 and not literal.satisfy
                if literal.pointer > -1:
                    look_ahead_continues.append(literal.tokens[literal.pointer + 1:])

        return look_ahead_continues

    def advance(self, word_id: int) -> 'ConstrainedHypothesis':
        obj = pickle.loads(pickle.dumps(self))

        for clause in obj.clauses:
            if clause.satisfy:
                continue

            clause.advance(word_id)
            if clause.satisfy:
                obj.orders.append(clause.idx)

        return obj

    def allowed(self) -> Set[int]:
        """

        :return: the tokens for next progress
        """
        allowed = set()

        for clause in self.clauses:
            if clause.satisfy:
                continue

            for literal in clause.literals:
                assert literal.pointer < len(literal.tokens) - 1 and not literal.satisfy
                allowed.add(literal.tokens[literal.pointer + 1])
                allowed.add(literal.tokens[0])

        return allowed







def init_batch(raw_constraints: List[List[List[List[int]]]],
               key_constraints: List[List[List[List[int]]]],
               beam_size: int,
               eos_id: Union[int, list]) -> List[Optional[ConstrainedHypothesis]]:
    """
    :param raw_constraints: The list of clause constraints.
    :param beam_size: The beam size.
    :param eos_id: The target-language vocabulary ID of the EOS symbol.
    :return: A list of ConstrainedHypothesis objects (shape: (batch_size * beam_size,)).
    """
    constraints_list = [None] * (len(raw_constraints) * beam_size)  # type: List[Optional[ConstrainedHypothesis]]
    for i, (raw_list, key_list) in enumerate(zip(raw_constraints, key_constraints)):
        hyp = ConstrainedHypothesis(raw_list, key_list, eos_id)
        idx = i * beam_size
        constraints_list[idx:idx + beam_size] = [copy.deepcopy(hyp) for _ in range(beam_size)]
    return constraints_list


class ConstrainedCandidate:
    """
    Object used to hold candidates for the beam in topk().

    :param row: The row in the scores matrix.
    :param col: The column (word ID) in the scores matrix.
    :param score: the associated accumulated score.
    :param hypothesis: The ConstrainedHypothesis containing information about met constraints.
    """

    __slots__ = ('row', 'col', 'score', 'hypothesis', 'rank')

    def __init__(self,
                 row: int,
                 col: int,
                 score: float,
                 hypothesis: ConstrainedHypothesis,
                 rank: float = None) -> None:
        self.row = row
        self.col = col
        self.score = score
        self.hypothesis = hypothesis
        self.rank = rank

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.row, self.col, self.score, self.hypothesis.num_met())


if __name__ == '__main__':
    clauses = [[[[3, 4, 5], [3, 4], [4, 5]], [[3, 4], [6], [7]]],
               [[[6], [6, 7], [6, 7, 8]], [[6, 9], [6, 4, 9]]],
               [[[3, 4, 5]], [[3, 4]], [[4, 5]]],
               [[[3, 4]], [[2, 3, 5]], [[6, 5]]],
               [[[2221], [9258], [3726], [2540], [6140]], [[4314], [18570]], [[1210], [6225], [2900], [4962]],
                [[6658, 6413], [17076], [6658, 84, 5700]], [[17076, 276], [17076], [6658, 84, 5700], [6658, 84, 12595]]]]

    constraints = init_batch(raw_constraints=clauses,
                             key_constraints=clauses,
                             beam_size=1,
                             eos_id=0)

    constraint = constraints[-1]
    print(constraint)
    print(constraints)
    print()
    for w in [2990,  2540,  6225, 18570,   290,  6658,  6413, 17076]:
        print(w)
        constraint = constraint.advance(w)
        print(constraint)
        print(constraint.allowed())
        print(constraint.met_order())
        print(constraint.clause_in_process())
        print()


