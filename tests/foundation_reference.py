"""Test-only constructive witnesses derived from the elementary contracts.

No production transition calls: this is separate from literal rule assertions.
Memory witnesses know the deck; they do not measure blind exploration.
"""
from games.skill_common import DIRS


def plan(p):
    kind = p.spec.game_id
    if kind=='bf01':
        return [i for i,(a,b) in enumerate(zip(p.state,p.target)) if a!=b]
    if kind=='bf02':
        # Explicit level contracts use topological indices, including both
        # branches before the final conjunction. No generic rule search.
        return list(range(p.n))
    if kind in ('bf03','bf05'):
        x,y=p.state; tx,ty=p.destination
        return [3 if tx>x else 2]*abs(tx-x)+[1 if ty>y else 0]*abs(ty-y)
    if kind=='bf04':
        return [i for i,(a,b) in enumerate(zip(p.state,p.target)) for _ in range((b-a)%4)]
    if kind=='bf06':
        # Locate the token required at the first reference position.
        left=p.state.index(p.target[0])
        return [1]*left if left<=p.n-left else [0]*(p.n-left)
    if kind=='bf07':
        if p.state[0]>p.target[0]:
            return [edge*2 for edge in range(p.n-1) for _ in range(p.state[0]-p.target[0])]
        return [edge*2+1 for edge in reversed(range(p.n-1)) for _ in range(p.state[-1]-p.target[-1])]
    if kind=='bf08':
        return [i for value in range(p.n) for i,v in enumerate(p.deck) if v==value]
    raise ValueError(kind)


def sdk_action(p,command):
    """Advertised arrows for movement; actual piece/arrow hitboxes otherwise."""
    if p.arrow_commands:
        return p.arrow_commands.index(command)+1,None
    x,y,w,h=p.button_rects()[command]
    return 6,{'x':x+w//2,'y':y+h//2}
