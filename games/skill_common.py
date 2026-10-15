"""Shared state, rendering and teacher-label contracts for synthetic micro-worlds."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

import numpy as np

LEVELS = 6
DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))


@dataclass(frozen=True)
class SkillSpec:
    game_id: str
    skill: str
    title: str
    context: str



class Canvas:
    """Small integer-pixel drawing helpers; all writes are clipped."""
    def __init__(self, palette: tuple[int, ...], size: int):
        self.pixels = np.full((64, 64), 5, dtype=np.int8)
        self.palette, self.size = palette, size

    def color(self, value: int) -> int:
        return self.palette[value % len(self.palette)]

    def rect(self, x, y, w, h, color, hollow=False):
        x, y, w, h = map(int, (x, y, w, h))
        if hollow:
            for xx, yy, ww, hh in ((x,y,w,1),(x,y+h-1,w,1),(x,y,1,h),(x+w-1,y,1,h)):
                self.rect(xx, yy, ww, hh, color)
        elif w > 0 and h > 0:
            x0,y0,x1,y1=max(0,x),max(0,y),min(64,x+w),min(64,y+h)
            if x1>x0 and y1>y0: self.pixels[y0:y1,x0:x1] = color

    def line(self, a, b, color=2):
        x, y = a; xx, yy = b
        count = max(abs(xx-x), abs(yy-y), 1)
        for u, v in zip(np.linspace(x,xx,count+1), np.linspace(y,yy,count+1)):
            self.rect(round(u), round(v), 1, 1, color)

    def token(self, x, y, value, hollow=False, size=None):
        size = self.size if size is None else size
        self.rect(x-size//2, y-size//2, size, size, self.color(value), hollow)

    def cross(self, x, y, value, radius=3):
        self.rect(x-radius,y,2*radius+1,1,self.color(value))
        self.rect(x,y-radius,1,2*radius+1,self.color(value))

    def arrow(self, x, y, dx, dy, color=0, radius=2):
        # At three pixels a shaft makes a symmetric plus. Use a chevron there.
        if radius>1: self.line((x-radius*dx,y-radius*dy),(x+radius*dx,y+radius*dy),color)
        for sign in (-1,1):
            self.line((x+radius*dx,y+radius*dy),
                      (x+sign*min(radius,2)*dy,y-sign*min(radius,2)*dx),color)

    def turn(self, x, y, clockwise=True, color=0):
        d=1 if clockwise else -1
        for a,b in (((-2,1),(-2,-2)),((-2,-2),(2,-2)),((2,-2),(2,2))):
            self.line((x+d*a[0],y+a[1]),(x+d*b[0],y+b[1]),color)
        self.line((x+2*d,y+2),(x,y),color)
        self.line((x+2*d,y+2),(x+3*d,y),color)

    def play(self, x, y, color=0):
        for i in range(4): self.rect(x+i,y-3+i,1,7-2*i,color)

    def wait(self, x, y, color=0):
        self.rect(x-2,y-2,1,5,color); self.rect(x+2,y-2,1,5,color)

    def cycle(self, x, y, count, current=None, step=7):
        """Ordered actual colour samples, forward arrows and an explicit return."""
        for v in range(count):
            self.token(x+step*v,y,v,size=3)
            if v<count-1: self.arrow(x+step*v+step//2,y,1,0,2,1)
            if current==v: self.rect(x+step*v-2,y-3,5,7,0,True)
        end=x+step*(count-1)
        self.line((end,y+3),(end,y+6),2)
        self.line((end,y+6),(x,y+6),2)
        self.arrow(x,y+4,0,-1,2,1)

    def slots(self, x, y, count, capacity, color, *, columns=6, hollow=False):
        """One separated cell per unit, including empty capacity and exact zero."""
        if not 0 <= count <= capacity:
            raise ValueError('quantity outside displayed capacity')
        for i in range(capacity):
            xx,yy=x+3*(i%columns),y+3*(i//columns)
            self.rect(xx,yy,2,2,color if i<count else 2)
            if hollow and i<count: self.rect(xx,yy,1,1,5)

    def label(self,x,y,index,color=0):
        letters=('010101111101101','110101110101110','111100100100111',
                 '110101101101110','111100110100111','111100110100100',
                 '111100101101111','101101111101101','111010010010111',
                 '001001001101111','101101110101101','100100100100111',
                 '101111111101101','101111111111101','111101101101111',
                 '111101111100100','111101101111001','110101110101101',
                 '111100111001111','111010010010010')
        for i,ch in enumerate(letters[index]):
            if ch=='1': self.rect(x+i%3,y+i//3,1,1,color)

    def digit(self, x, y, n, color=0):
        glyphs = ('111101101101111','010110010010111','111001111100111',
                  '111001111001111','101101111001001','111100111001111',
                  '111100111101111','111001001001001','111101111101111',
                  '111101111001111')
        for i, ch in enumerate(glyphs[int(n) % 10]):
            if ch == '1': self.rect(x+i%3,y+i//3,1,1,color)



def condition(name: str, current: Any, target: Any, *, monotonic=False, depends_on=()):
    return {'id': name, 'achieved': current == target, 'current': current,
            'target': target, 'monotonic': monotonic, 'depends_on': list(depends_on)}



class Puzzle:
    context = ''
    arrow_commands: tuple[int, ...] = ()
    run_command: int | None = None

    def __init__(self, spec: SkillSpec, seed: int, level: int, *, appearance_seed: int | None = None):
        if type(seed) is not int or type(level) is not int or not 1 <= level <= LEVELS:
            raise ValueError('seed must be an integer and level must be in 1..6')
        self.spec, self.seed, self.level = spec, seed, level
        self.rng = random.Random(f'skill-logic-v1:{spec.skill}:{seed}:{level}')
        cosmetic = random.Random(f'skill-look-v1:{spec.skill}:{seed if appearance_seed is None else appearance_seed}:{level}')
        self.palette = tuple(cosmetic.sample(range(6,16), 10))
        self.size = cosmetic.randint(3,5)
        self.offset = (cosmetic.randint(0,2), cosmetic.randint(0,2))
        self.commands: tuple[str, ...] = ()
        self.state: tuple = ()
        self.target: tuple = ()
        self.history: list[tuple] = []
        self.steps, self.lost, self.won = 0, False, False
        self.event = ''
        self.transient_frames: list[np.ndarray] = []
        self.budget = 96 + 32 * (level - 1)
        self.setup()
        self.initial_state = self.state
        if self.solved(self.state):
            raise ValueError(f'{spec.skill}/{seed}/{level}: generator produced a solved start')

    def setup(self):
        raise NotImplementedError

    def transition(self, state: tuple, command: int) -> tuple | None:
        """Pure transition; None means a fatal action, identity a blocked action."""
        raise NotImplementedError

    def projection(self, state):
        return state

    def subgoals(self, state=None):
        values = self.projection(self.state if state is None else state)
        return [condition(f'target-{i}', a, b) for i,(a,b) in enumerate(zip(values,self.target))]

    def solved(self, state):
        goals = self.subgoals(state)
        return bool(goals) and all(g['achieved'] for g in goals)

    def target_is_new(self, projected):
        return projected != self.projection(self.state)

    def reachable_target(self, steps=None):
        """Construct a reachable target without retaining an oracle action sequence."""
        state = self.state
        candidates = []
        for _ in range(steps or 2*self.level-1):
            options = [self.transition(state, i) for i in range(len(self.commands))]
            options = [s for s in options if s is not None and s != state]
            if not options: break
            productive = [s for s in options if self.target_is_new(self.projection(s))]
            state = self.rng.choice(productive or options)
            if self.target_is_new(self.projection(state)): candidates.append(state)
        if not candidates:
            raise ValueError(f'No reachable target for {self.spec.skill}')
        self.target = tuple(self.projection(candidates[-1]))

    def act(self, command: int):
        if self.won or self.lost: return
        if type(command) is not int or not 0 <= command < len(self.commands):
            raise ValueError('invalid command')
        self.history.append(self.state)
        self.steps += 1
        nxt = self.transition(self.state, command)
        self.transient_frames = self.transition_frames(self.state,command,nxt)
        self.event = 'blocked' if nxt == self.state else 'changed'
        if nxt is None:
            self.lost, self.event = True, 'hazard'
        else:
            self.state = nxt
            self.won = self.solved(nxt)
            self.lost = not self.won and self.steps >= self.budget

    def transition_frames(self,before,command,after):
        return []

    def render_state(self,state):
        from copy import copy
        view=copy(self)
        view.state=state
        return view.render()

    def miss(self):
        if not (self.won or self.lost):
            self.steps += 1
            self.lost = self.steps >= self.budget
            self.event = 'miss'
            self.transient_frames = []

    def undo(self):
        # Terminal states require RESET, as in the SDK. Undo never refunds budget.
        if self.won or self.lost: return
        self.steps += 1
        if self.history: self.state = self.history.pop()
        self.event = 'undo'
        self.transient_frames = []
        self.lost = self.steps >= self.budget

    def button_rects(self):
        n = len(self.commands)
        return [(1+8*i, 53, 7, 7) for i in range(n)]

    def draw_controls(self,c):
        for i,(x,y,w,h) in enumerate(self.button_rects()):
            c.rect(x,y,w,h,2,True)
            self.draw_control(c,i,x+w//2,y+h//2)

    def draw_control(self,c,command,x,y):
        if command in self.arrow_commands:
            c.arrow(x,y,*DIRS[self.arrow_commands.index(command)])
        else:
            raise NotImplementedError(f'Missing pictorial control: {self.spec.context}/{command}')

    def click(self, x, y):
        if type(x) is not int or type(y) is not int or not (0<=x<64 and 0<=y<64):
            self.miss(); return
        for i,(bx,by,w,h) in enumerate(self.button_rects()):
            if bx <= x < bx+w and by <= y < by+h:
                self.act(i); return
        self.miss()

    def draw(self, c: Canvas):
        live, goal = self.projection(self.state), self.target
        for i, (a,b) in enumerate(zip(live,goal)):
            x = 6 + (i % 8)*7; y = 15 + (i//8)*16
            c.token(x,y,int(b),True,size=7)
            c.token(x,y+8,int(a))

    def render(self):
        c = Canvas(self.palette,self.size)
        self.draw(c)
        if not getattr(self, 'board_controls', False): self.draw_controls(c)
        goals = self.subgoals()
        for i,g in enumerate(goals[:30]):
            c.rect(2+2*i,1,1,2,0 if g['achieved'] else 2)
        left = math.ceil(60*max(0,self.budget-self.steps)/self.budget)
        c.rect(2,62,60,1,2); c.rect(2,62,left,1,0)
        return c.pixels

    def get_subgoals(self):
        from copy import deepcopy
        return deepcopy(self.subgoals())

    def get_training_info(self):
        # Fresh JSON-compatible data: no transition tables, programs or solution.
        return {'schema_version':1, 'training_only':True, 'source':'synthetic',
                'game_id':self.spec.game_id+'-v1', 'skill':self.spec.skill,
                'context':self.spec.context, 'seed':self.seed, 'level':self.level,
                'steps':self.steps, 'budget':self.budget,
                'difficulty':{name:getattr(self,name) for name in ('n','m','mod','limit','depth','d','slots','period') if hasattr(self,name)},
                'subgoals':self.get_subgoals()}



