"""Elementary synthetic mechanics, isolated from the later skill profiles.

Only native current subgoals leave the detached teacher interface. Witnesses
belong to tests; generators store instances, never action plans.
"""
from __future__ import annotations

import math

from games.skill_common import DIRS, Puzzle, condition


def arrow(c, x, y, dx, dy, color=0):
    """A direction, drawn without a symbolic opcode or alphabetic label."""
    c.line((x-3*dx, y-3*dy), (x+3*dx, y+3*dy), color)
    for sign in (-1, 1):
        c.line((x+3*dx, y+3*dy),
               (x+dx+sign*2*dy, y+dy-sign*2*dx), color)


def turn_arrow(c, x, y, clockwise=True):
    points = [(x+round(4*math.cos(t)), y+round(4*math.sin(t)))
              for t in [i*math.pi/12 for i in range(3, 22)]]
    if not clockwise:
        points = [(2*x-u, v) for u, v in points]
    for a, b in zip(points, points[1:]):
        c.line(a, b, 0)
    u, v = points[-1]
    c.line((u, v), (u-3 if clockwise else u+3, v), 0)
    c.line((u, v), (u, v-3), 0)


class Basic(Puzzle):
    board_controls = True
    direct_controls = True

    def centers(self, count):
        # Appearance changes placement, not answer or logical ordering.
        ox, oy = self.offset
        xs = {1: (31,), 2: (18, 44), 3: (11, 31, 51)}[count]
        return [(x+ox-1, (29 if self.level%2 else 35)+oy) for x in xs]


class MatchColor(Basic):
    def setup(self):
        self.n = (self.level+1)//2
        self.state = tuple(self.rng.randrange(2) for _ in range(self.n))
        self.target = tuple(1-v for v in self.state)
        self.commands = tuple('Tap a filled tile' for _ in range(self.n))

    def transition(self, state, command):
        return tuple(1-v if i == command else v for i, v in enumerate(state))

    def button_rects(self):
        side = self.size+5
        return [(x-side//2, y, side, side) for x, y in self.centers(self.n)]

    def draw(self, c):
        for i, (x, y, w, h) in enumerate(self.button_rects()):
            c.rect(x-1, y-15, w+2, h+2, c.color(self.target[i]), True)
            c.line((x+w//2, y-3), (x+w//2, y-1), 2)
            c.rect(x, y, w, h, c.color(self.state[i]))


class Unlock(Basic):
    def setup(self):
        self.parents = ( ((),), ((), (0,)), ((), ()),
                         ((), (0,), (1,)), ((), (0,), (), (2,)),
                         ((), (0,), (), (2,), (1, 3)) )[self.level-1]
        self.n = len(self.parents)
        self.state = (False,)*self.n
        self.target = (True,)*self.n
        self.commands = ('Open a connected switch',)*self.n

    def transition(self, state, command):
        if not all(state[i] for i in self.parents[command]):
            return state
        return tuple(True if i == command else v for i, v in enumerate(state))

    def subgoals(self, state=None):
        state = self.state if state is None else state
        return [condition(f'switch-{i}', v, True, monotonic=True,
                          depends_on=[f'switch-{j}' for j in self.parents[i]])
                for i, v in enumerate(state)]

    def positions(self):
        layouts = (((23,30),), ((15,30),(36,30)), ((23,18),(23,43)),
                   ((9,30),(27,30),(45,30)),
                   ((12,18),(34,18),(12,43),(34,43)),
                   ((9,18),(28,18),(9,43),(28,43),(48,30)))
        ox, oy = self.offset
        return [(x+ox, y+oy-1) for x,y in layouts[self.level-1]]

    def button_rects(self):
        s = self.size+4
        return [(x-s//2, y-s//2, s, s) for x,y in self.positions()]

    def draw(self, c):
        positions = self.positions()
        leaves = set(range(self.n)) - {p for parents in self.parents for p in parents}
        for i, parents in enumerate(self.parents):
            for p in parents:
                c.line(positions[p], positions[i], c.color(0) if self.state[p] else 2)
        for i in leaves:
            x,y = positions[i]
            c.line((x,y), (x+10,y), c.color(0) if self.state[i] else 2)
            c.rect(x+8,y-2,5,5,c.color(0),not self.state[i])
        for i, (x,y,w,h) in enumerate(self.button_rects()):
            c.rect(x-1,y-1,w+2,h+2,5)
            c.rect(x,y,w,h,c.color(0),True)
            if self.state[i]:
                c.line((x+2,y+h//2),(x+w//2,y+h-2),0)
                c.line((x+w//2,y+h-2),(x+w-2,y+2),0)
            else:
                # Closed crossbar; a gap in the top shows an enabled switch.
                c.rect(x+2,y+h//2,w-4,2,2)
                if all(self.state[p] for p in self.parents[i]):
                    c.rect(x+w//2-1,y,3,1,5)


class Move(Basic):
    arrow_commands = (0,1,2,3)
    mirror = False

    def setup(self):
        self.width, self.height = (3,5) if self.mirror else (5,5)
        self.distance = (1,2,3,4,5,6)[self.level-1] if self.mirror else (1,2,2,3,4,6)[self.level-1]
        pairs = []
        cells = [(x,y) for x in range(self.width) for y in range(self.height)]
        for a in cells:
            for b in cells:
                dx,dy = abs(a[0]-b[0]),abs(a[1]-b[1])
                if dx+dy != self.distance:
                    continue
                if not self.mirror and ((self.level==2 and dx and dy) or
                                        (self.level==3 and not (dx and dy))):
                    continue
                pairs.append((a,b))
        self.state, self.destination = self.rng.choice(pairs)
        self.target = self.project(self.destination)
        self.commands = ('Move up','Move down','Move left','Move right')

    def project(self, state):
        x,y = state
        return (self.width-1-x,y) if self.mirror else state

    def projection(self, state):
        return self.project(state)

    def subgoals(self, state=None):
        return [condition('reflection' if self.mirror else 'object',
                          self.project(self.state if state is None else state), self.target)]

    def transition(self, state, command):
        dx,dy = DIRS[command]
        x,y = state[0]+dx,state[1]+dy
        return (x,y) if 0 <= x < self.width and 0 <= y < self.height else state

    def button_rects(self):
        # Visible arrows share the same mapping as keyboard and external D-pad.
        return [(4+i*15,51,11,9) for i in range(4)]

    def point(self, state):
        x,y = state
        return (8+7*x+self.offset[0], 10+8*y+self.offset[1]) if self.mirror else (13+9*x+self.offset[0]-1, 10+8*y+self.offset[1])

    def draw(self,c):
        for x in range(self.width):
            for y in range(self.height):
                u,v = self.point((x,y))
                c.rect(u,v,1,1,2)
                if self.mirror: c.rect(63-u,v,1,1,2)
        tx,ty = self.point(self.destination)
        x,y = self.point(self.state)
        s = self.size+2
        if self.mirror:
            c.rect(31,6,2,41,2)
            tx = 63-tx
            c.token(63-x,y,0,size=self.size)
            c.token(63-x,y,0,True,size=s+2)
            c.token(x,y,0,size=s)
            c.rect(x,y,1,1,0)  # controlled solid with center; outlined reflection
        else:
            c.token(x,y,0,size=s)
        c.token(tx,ty,0,True,size=s+2)
        for (bx,by,w,h),(dx,dy) in zip(self.button_rects(),DIRS):
            c.rect(bx,by,w,h,2,True)
            arrow(c,bx+w//2,by+h//2,dx,dy)


class Reflect(Move):
    mirror = True


class Rotate(Basic):
    def setup(self):
        self.n = 1 if self.level <= 4 else 2
        self.shapes = (((0,0),(1,0),(0,1)), ((0,0),(1,0),(-1,0),(-1,1)))
        self.shape = 0 if self.level <= 3 else 1
        self.state = tuple(self.rng.randrange(4) for _ in range(self.n))
        turns = ((1,), (2,), (3,), (1,), (1,1), (1,2))[self.level-1]
        self.target = tuple((v+d)%4 for v,d in zip(self.state,turns))
        self.commands = ('Tap a piece to turn clockwise',)*self.n

    def transition(self, state, command):
        return tuple((v+1)%4 if i==command else v for i,v in enumerate(state))

    def positions(self):
        return [(31+self.offset[0]-1,31+self.offset[1])] if self.n==1 else [(16+self.offset[0]-1,31+self.offset[1]),(46+self.offset[0]-1,31+self.offset[1])]

    def button_rects(self):
        return [(x-11,y-11,23,23) for x,y in self.positions()]

    def cells(self, orientation):
        cells = self.shapes[self.shape]
        for _ in range(orientation):
            cells = tuple((-y,x) for x,y in cells)
        return cells

    def draw(self,c):
        for i,(x,y) in enumerate(self.positions()):
            turn_arrow(c,x,y-19)
            for dx,dy in self.cells(self.target[i]):
                c.rect(x+7*dx-3,y+7*dy-3,7,7,c.color(i),True)
            for dx,dy in self.cells(self.state[i]):
                c.token(x+7*dx,y+7*dy,i,size=self.size)
            c.rect(x,y,1,1,0)


class Loop(Basic):
    def setup(self):
        self.n = 2+(self.level-1)//2
        values = list(range(self.n)); self.rng.shuffle(values)
        self.state = tuple(values)
        offset = self.rng.choice((-1,1)) if self.n<4 else self.rng.choice((-2,-1,1,2))
        self.target = self.state[offset:]+self.state[:offset]
        self.commands = ('Turn loop clockwise','Turn loop counterclockwise')

    def transition(self,state,command):
        return state[-1:]+state[:-1] if command==0 else state[1:]+state[:1]

    def button_rects(self):
        return [(12,49,15,11),(37,49,15,11)]

    def draw(self,c):
        for cx,cy,values,reference in ((18+self.offset[0]-1,29+self.offset[1],self.state,False),
                                     (47+self.offset[0]-1,29+self.offset[1],self.target,True)):
            points = [(cx+round(10*math.sin(i*math.pi/24)),cy-round(10*math.cos(i*math.pi/24))) for i in range(49)]
            for a,b in zip(points,points[1:]): c.line(a,b,2)
            for i,value in enumerate(values):
                x=cx+round(10*math.sin(2*math.pi*i/self.n))
                y=cy-round(10*math.cos(2*math.pi*i/self.n))
                c.token(x,y,value,reference,size=self.size+3)
        for i,(x,y,w,h) in enumerate(self.button_rects()):
            c.rect(x,y,w,h,2,True)
            turn_arrow(c,x+w//2,y+h//2,clockwise=i==0)


class Transfer(Basic):
    def setup(self):
        self.n = 2 if self.level<=4 else 3
        self.capacity = (2,2,4,4,4,4)[self.level-1]
        self.transfers = (1,2,3,4,4,6)[self.level-1]
        if self.n==2:
            start,target = (self.transfers,0),(0,self.transfers)
        else:
            amount = self.transfers//2
            start,target = (amount,0,0),(0,0,amount)
        if self.rng.randrange(2): start,target = start[::-1],target[::-1]
        self.state,self.target = start,target
        self.commands = tuple('Transfer one unit right' if d==0 else 'Transfer one unit left'
                              for _ in range(self.n-1) for d in range(2))

    def transition(self,state,command):
        edge,d = divmod(command,2)
        a,b = (edge,edge+1) if d==0 else (edge+1,edge)
        if state[a]==0 or state[b]>=self.capacity: return state
        result = list(state); result[a]-=1; result[b]+=1
        return tuple(result)

    def positions(self):
        return [16+self.offset[0],45+self.offset[0]] if self.n==2 else [10+self.offset[0],30+self.offset[0],50+self.offset[0]]

    def button_rects(self):
        xs = self.positions()
        return [((a+b)//2-5,48+7*d,11,7) for a,b in zip(xs,xs[1:]) for d in range(2)]

    def draw(self,c):
        for i,x in enumerate(self.positions()):
            w=self.size+3
            for slot in range(self.capacity):
                y=39-slot*7
                c.rect(x-w//2,y,w,6,2,True)
                if slot<self.state[i]: c.rect(x-w//2+1,y+1,w-2,4,c.color(0))
                # Hollow target slots beside each tank: no numeric reading.
                if slot<self.target[i]: c.rect(x+w//2+2,y+1,3,4,c.color(0),True)
        for i,(x,y,w,h) in enumerate(self.button_rects()):
            c.rect(x,y,w,h,2,True)
            arrow(c,x+w//2,y+h//2,1 if i%2==0 else -1,0)


class Pair(Basic):
    def setup(self):
        self.n = (self.level+1)//2
        deck = list(range(self.n))*2; self.rng.shuffle(deck)
        self.deck = tuple(deck)
        self.state = (0,-1,-1)
        self.target = (True,)*self.n
        self.commands = ('Reveal a card',)*(2*self.n)

    def transition(self,state,command):
        matched,opened,_ = state
        if matched & (1<<command) or opened==command: return state
        if opened<0: return matched,command,-1
        if self.deck[opened]==self.deck[command]:
            return matched | (1<<opened) | (1<<command),-1,-1
        return matched,-1,-1

    def subgoals(self,state=None):
        matched,_,_ = self.state if state is None else state
        return [condition(f'pair-{i}',all(bool(matched & (1<<j)) for j,v in enumerate(self.deck) if v==i),True,monotonic=True) for i in range(self.n)]

    def transition_frames(self,before,command,after):
        if before[1]>=0 and before[1]!=command and not before[0] & (1<<command):
            # Eight native frames keep both faces perceivable (~640 ms in play).
            return [self.render_state((before[0],before[1],command)) for _ in range(8)]
        return []

    def button_rects(self):
        w,h = self.size+7,self.size+10
        cols = 2 if self.n<3 else 3
        ox,oy=self.offset
        return [(10+19*(i%cols)+ox,(12 if self.level%2 else 16)+23*(i//cols)+oy,w,h) for i in range(2*self.n)]

    def draw(self,c):
        for i,(x,y,w,h) in enumerate(self.button_rects()):
            matched = bool(self.state[0] & (1<<i))
            opened = i in self.state[1:]
            c.rect(x,y,w,h,0 if opened else 2,True)
            if matched or opened:
                c.token(x+w//2,y+h//2,self.deck[i],size=self.size+2)
                if matched: c.rect(x+2,y+h-3,w-4,1,0)
            else:
                c.rect(x+3,y+4,w-6,h-8,c.color(9),True)
