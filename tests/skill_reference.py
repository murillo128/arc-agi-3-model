"""Test-only search and explicit teaching plans. Never imported by production."""
from collections import deque
from itertools import product


def expand(codes, depth=1, recursive=False):
    """Independent declarative expansion, not the production stack interpreter."""
    def call(d):
        return (0,) * d + (1,) * d
    return sum((call(c) if recursive else
                () if c == 0 else (c - 1,) if c < 3 else call(depth)
                for c in codes), ())


def plan(puzzle, limit=150_000):
    """Find a witness using actual transitions; literal tests own rule correctness."""
    p = puzzle
    kind = p.spec.context
    if kind == 'memory':
        return [i for value in range(p.n) for i, v in enumerate(p.deck) if v == value]
    if kind == 'rules':
        answer = p.phrase
        for table in p.maps:
            answer = tuple(table[value] for value in answer)
        return [i for i, value in enumerate(answer)
                for _ in range((value-p.state[i]) % p.k)]
    if kind == 'sigils':
        return [bit for i, value in enumerate(p.target) for _ in range(value)
                for bit in range(9) if p.patterns[i] & (1 << bit)]
    if kind == 'program':
        codes = next(c for c in product(range(4), repeat=p.n)
                     if expand(c, p.depth, p.recursive) == p.target)
        return [2*i+bit for i,c in enumerate(codes) for bit in range(2)
                if c & (1 << bit)] + [p.run_command]
    if kind == 'centroid':
        if p.level == 1:
            return [1]
        direction = 0 if p.target[0] == 0 else 1
        return [2*i+direction for i, value in enumerate(p.state)
                for _ in range(value if direction == 0 else p.limit-value)]
    if kind == 'merge':
        board = list(p.state[0]); commands = []
        for tier, required in enumerate(p.target):
            while board.count(tier) > required:
                indices = [i for i,v in enumerate(board) if v == tier]
                if len(indices) < 2:
                    raise AssertionError('Generated target has incompatible mass')
                a,b = indices[:2]
                commands += [a,b]; board[a] = -1; board[b] += 1
        return commands
    if kind == 'ghost':
        commands = [2 if p.plates[0][0]<0 else 3] * p.d + [4]
        if p.slots == 2:
            commands += [3] * p.d + [4]
        return commands + [5] * (p.d-1) + [1] * p.exit_depth
    if kind == 'pegs':
        # Search complete hops, not the irrelevant selected-peg permutations.
        queue = deque([p.state]); parents = {p.state: None}
        while queue:
            state = queue.popleft()
            if p.solved(state):
                path = []
                while parents[state] is not None:
                    state, pair = parents[state]; path[:0] = pair
                return path
            board,_ = state
            for a,v in enumerate(board):
                if v != 1: continue
                x,y = a % p.n, a // p.n
                for dx,dy in ((0,1),(0,-1),(-1,0),(1,0)):
                    if not (0 <= x+2*dx < p.n and 0 <= y+2*dy < p.n): continue
                    b = (y+2*dy)*p.n+x+2*dx
                    selected = p.transition(state,a)
                    nxt = p.transition(selected,b)
                    if nxt == selected or nxt in parents: continue
                    parents[nxt] = state,[a,b]; queue.append(nxt)
        raise AssertionError('No legal peg solution')
    queue = deque([p.state]); parents = {p.state: None}
    while queue:
        state = queue.popleft()
        if p.solved(state):
            path = []
            while parents[state] is not None:
                state, command = parents[state]; path.append(command)
            return path[::-1]
        for command in range(len(p.commands)):
            nxt = p.transition(state, command)
            if nxt is None or nxt in parents: continue
            parents[nxt] = state,command; queue.append(nxt)
            if len(parents) > limit:
                raise AssertionError(f'Search exceeded {limit} states: {p.spec.skill}/{p.seed}/{p.level}')
    raise AssertionError(f'No solution: {p.spec.skill}/{p.seed}/{p.level}')
