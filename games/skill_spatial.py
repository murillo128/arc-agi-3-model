"""Original synthetic mechanics; no SDK or evaluated-policy dependencies."""
from __future__ import annotations

from collections import deque

from games.skill_common import Canvas, DIRS, Puzzle, condition


class Transform(Puzzle):
    arrow_commands = (0,1,2,3)
    native_commands = {5:4}
    def setup(self):
        self.m = 3+self.level//2
        self.pattern = ((0,0),) if self.level == 1 else tuple((i%2,i//2) for i in range(min(4,self.level)))
        self.mirror = self.spec.skill in ('simetria','razonamiento_inverso')
        self.moving_axis = self.spec.skill == 'marcos_de_referencia'
        operation = 'Cycle mirror position' if self.moving_axis else 'Rotate clockwise'
        self.commands=('Up','Down','Left','Right',operation,'Reflect the piece')
        self.state=(1,1,0,0,self.m//2)
        self.reachable_target(2*self.level-1)

    def target_is_new(self,projected):
        return not set(projected).issubset(set(self.cells(self.state)))

    def cells(self,state):
        x,y,r,flip,axis=state; cells=[]
        for dx,dy in self.pattern:
            if flip: dx=-dx
            for _ in range(r): dx,dy=-dy,dx
            px,py=x+dx,y+dy
            cells.append((px,py))
            if self.mirror or self.moving_axis: cells.append((2*axis-px,py))
        return tuple(sorted(set(cells)))

    def projection(self,state):
        return self.cells(state)

    def transition(self,state,command):
        x,y,r,f,a=state
        if command < 4:
            dx,dy=DIRS[command]; x+=dx; y+=dy
        elif command == 4:
            if self.moving_axis:
                # One direct action cycles through legal mirror positions.
                for step in range(1,self.m+2):
                    candidate=(x,y,r,f,(a+step)%(self.m+1))
                    if all(0<=u<=self.m and 0<=v<=self.m for u,v in self.cells(candidate)):
                        return candidate
                return state
            r=(r+1)%4
        else: f=1-f
        s=(x,y,r,f,a)
        return s if 0<=a<=self.m and all(0<=u<=self.m and 0<=v<=self.m for u,v in self.cells(s)) else state

    def subgoals(self,state=None):
        occupied=set(self.cells(self.state if state is None else state))
        return [condition(f'mark-{i}',p in occupied,True) for i,p in enumerate(self.target)]

    def draw(self,c):
        unit=5; ox,oy=8+self.offset[0],12+self.offset[1]
        c.rect(ox-3,oy-3,(self.m+1)*unit,(self.m+1)*unit,2,True)
        if self.mirror or self.moving_axis:
            x=ox+self.state[4]*unit; c.line((x,oy-2),(x,oy+self.m*unit),c.color(3))
        for x,y in self.target: c.cross(ox+x*unit,oy+y*unit,0,2)
        for x,y in self.cells(self.state): c.token(ox+x*unit,oy+y*unit,1)

    def piece_rects(self):
        half = self.size//2
        return [(8+self.offset[0]+x*5-half,12+self.offset[1]+y*5-half,
                 self.size,self.size) for x,y in self.cells(self.state)]

    def click_commands(self):
        # A reflection acts on the piece itself, rather than a detached icon.
        return [(5,rect) for rect in self.piece_rects()]


class Twins(Puzzle):
    arrow_commands=(0,1,2,3)
    def setup(self):
        self.m=2+self.level
        self.commands=('Up together','Down together','Left / right','Right / left')
        delta=0 if self.level==1 else min(2,self.level-1)
        self.state=(self.rng.randrange(max(1,self.m//3)),0,
                    self.m-self.rng.randrange(max(1,self.m//3)),delta)
        # The open top boundary is a guaranteed symmetry-breaking tool. Blocks
        # lie strictly below it, preserving a route to a common row and meeting.
        self.walls=frozenset((self.m//2,y) for y in range(2,self.m) if y%2==0)
        self.target=(True,)

    def transition(self,state,command):
        x,y,u,v=state; dx,dy=DIRS[command]
        def moved(a,b,d):
            p=(a+d,b+dy)
            return p if 0<=p[0]<=self.m and 0<=p[1]<=self.m and p not in self.walls else (a,b)
        p,q=moved(x,y,dx),moved(u,v,-dx)
        if p==(u,v) and q==(x,y): q=p
        return (*p,*q)

    def projection(self,state): return (state[:2]==state[2:],)

    def draw(self,c):
        unit=4 if self.m>=7 else 5; ox,oy=9+self.offset[0],9+self.offset[1]
        c.rect(ox-3,oy-3,(self.m+1)*unit+1,(self.m+1)*unit+1,2,True)
        for x,y in self.walls: c.rect(ox+x*unit-2,oy+y*unit-2,5,5,2)
        for k in (0,2): c.token(ox+self.state[k]*unit,oy+self.state[k+1]*unit,k//2)
        c.token(57,18,0,True,size=7); c.token(57,18,1,size=3)


class Centroid(Puzzle):
    native_commands={1:2,2:3,3:0,4:1}
    def setup(self):
        self.n=min(2,self.level)
        self.limit=3+self.level
        self.commands=tuple(f'{"Left" if i%2==0 else "Right"} arm {i//2+1}' for i in range(2*self.n))
        self.state=(self.limit//2,)*self.n
        self.target=(self.state[0]+1,) if self.level==1 else (self.rng.choice((0,self.n*self.limit)),)

    def transition(self,state,command):
        s=list(state); i=command//2
        s[i]=max(0,min(self.limit,s[i]+(-1 if command%2==0 else 1)))
        return tuple(s)

    def projection(self,state): return (sum(state),)

    def subgoals(self,state=None):
        s=self.state if state is None else state
        return [condition('centroid-numerator',sum(s),self.target[0])]

    def draw(self,c):
        unit=4; body=(7+round(sum(self.state)*unit/self.n),45)
        c.cross(7+round(self.target[0]*unit/self.n),45,0,3)
        for i,x in enumerate(self.state):
            p=(7+x*unit,10+i*10); c.line(p,body,2)
            c.line((p[0]+3,p[1]),(46,p[1]),c.color(1+i))
            c.token(*p,1+i)
        c.token(*body,1,size=self.size+2)


class Linkage(Puzzle):
    native_commands={1:2,2:3,3:0,4:1}
    def setup(self):
        self.n=min(2,1+self.level//2)
        self.limit=1+self.level
        self.staging=self.spec.skill=='subobjetivos_no_monotonos'
        self.commands=tuple(f'{"Retract" if i%2==0 else "Extend"} rod {i//2+1}' for i in range(2*self.n))
        self.state=(1,)*self.n
        if self.staging and self.n>1:
            self.target=(1,)+tuple(self.rng.randint(2,self.limit) for _ in range(self.n-1))
        else: self.target=tuple(self.rng.choice((0,*range(2,self.limit+1))) for _ in range(self.n))

    def transition(self,state,command):
        s=list(state); i=command//2
        # A child can only be adjusted while its parent is retracted. Restoring
        # the parent then moves every child endpoint, including solved markers.
        if self.staging and i and s[i-1] != 0: return state
        s[i]=max(0,min(self.limit,s[i]+(-1 if command%2==0 else 1)))
        return tuple(s)

    def points(self,state):
        x,y=6,9; points=[]
        unit=2 if self.limit>4 else 3
        for i,v in enumerate(state):
            if i%2: y+=unit*(v+1)
            else: x+=unit*(v+1)
            points.append((x,y))
        return points

    def draw(self,c):
        start=(6,9)
        for i,p in enumerate(self.points(self.target)):
            c.cross(*p,i,2)
        for i,p in enumerate(self.points(self.state)):
            c.line(start,p,c.color(i)); c.token(*p,i); start=p
        if self.staging:
            # A parent at minimum length enables its child. Keep this clue on
            # the actual endpoints, rather than connecting removed buttons.
            for i,(x,y) in enumerate(self.points(self.state)[1:],1):
                c.rect(x+3,y-1,2,3,0 if self.state[i-1]==0 else 2)


class Maze(Puzzle):
    arrow_commands=(0,1,2,3)
    native_commands={5:4}
    def setup(self):
        self.commands=('Up','Down','Left','Right','Toggle bridge')
        if self.level==1:
            self.n=2; self.edges={frozenset((0,1))}; self.exit=1
        else:
            self.n=3+(self.level-2)//2
            visited={0}; stack=[0]; self.edges=set()
            while stack:
                p=stack[-1]; x,y=p%self.n,p//self.n
                options=[(y+dy)*self.n+x+dx for dx,dy in DIRS
                         if 0<=x+dx<self.n and 0<=y+dy<self.n and (y+dy)*self.n+x+dx not in visited]
                if not options: stack.pop(); continue
                q=self.rng.choice(options); self.edges.add(frozenset((p,q))); visited.add(q); stack.append(q)
            parent={0:None}; queue=deque([0])
            while queue:
                p=queue.popleft()
                for e in sorted(self.edges,key=lambda e:sorted(e)):
                    if p in e:
                        q=next(v for v in e if v!=p)
                        if q not in parent: parent[q]=p; queue.append(q)
            self.exit=next(reversed(parent))
        parent={0:None}; queue=deque([0])
        while queue:
            p=queue.popleft()
            for e in sorted(self.edges,key=lambda e:sorted(e)):
                if p in e:
                    q=next(v for v in e if v!=p)
                    if q not in parent: parent[q]=p; queue.append(q)
        path=[]; p=self.exit
        while p is not None: path.append(p); p=parent[p]
        path.reverse()
        key_context=self.spec.skill=='planificacion_en_estado_ampliado'
        self.keys={}; self.doors={}
        if key_context and self.level>=2:
            for j,k in enumerate(sorted(set(max(2,len(path)*i//(min(3,self.level-1)+1))
                                            for i in range(1,min(3,self.level-1)+1)))):
                if k<len(path):
                    self.keys[path[k-2]]=j
                    self.doors[frozenset((path[k-1],path[k]))]=j
        bridge_context=self.spec.skill in ('topologia_dinamica','invariantes_de_seguridad')
        self.bridge=path[len(path)//2] if bridge_context and self.level>1 else -1
        initial_keys=1<<self.keys[0] if 0 in self.keys else 0
        self.state=(0,initial_keys,0)
        self.target=(self.exit,)
        if self.spec.skill=='recursos_limitados': self.budget=max(4,2*self.n*self.n+2*self.level)

    def transition(self,state,command):
        p,keys,bridge=state
        if command==4:
            if p==self.bridge and bridge: return None
            return (p,keys,1-bridge)
        x,y=p%self.n,p//self.n; dx,dy=DIRS[command]
        if not (0<=x+dx<self.n and 0<=y+dy<self.n): return state
        q=(y+dy)*self.n+x+dx; edge=frozenset((p,q))
        if edge not in self.edges or (q==self.bridge and not bridge): return state
        if edge in self.doors and not (keys & (1<<self.doors[edge])): return state
        if q in self.keys: keys |= 1<<self.keys[q]
        return (q,keys,bridge)

    def subgoals(self,state=None):
        p,keys,_=self.state if state is None else state
        return [condition(f'key-{i}',bool(keys&(1<<i)),True,monotonic=True) for i in sorted(set(self.doors.values()))] + [condition('exit',p,self.exit)]

    def draw(self,c):
        def point(p): return (8+(p%self.n)*8+self.offset[0],10+(p//self.n)*8+self.offset[1])
        for e in self.edges:
            a,b=sorted(e); c.line(point(a),point(b),2)
            if e in self.doors and not self.state[1]&(1<<self.doors[e]):
                x,y=point(a); u,v=point(b); c.token((x+u)//2,(y+v)//2,self.doors[e]+2,True)
        for p in range(self.n*self.n): c.rect(point(p)[0],point(p)[1],1,1,2)
        if self.bridge>=0: c.token(*point(self.bridge),4,hollow=not self.state[2],size=7)
        for p,k in self.keys.items():
            if not self.state[1]&(1<<k): c.cross(*point(p),k+2,2)
        c.cross(*point(self.exit),0,3); c.token(*point(self.state[0]),1)


class Boxes(Puzzle):
    arrow_commands=(0,1,2,3)
    def setup(self):
        self.n=4+self.level//3
        self.commands=('Up','Down','Left','Right')
        self.impulse=self.spec.skill=='dinamica_por_contacto'
        self.count=1 if self.level<3 else 2
        self.relevant=1 if self.spec.skill=='relevancia_del_objetivo' else self.count
        row=self.n//2
        self.state=(row*self.n,tuple(row*self.n+1+i*self.n for i in range(self.count)))
        # A guaranteed first contact gives a nontrivial goal even when a later
        # random walk mostly explores the player's free cells.
        pushed=self.transition(self.state,3)
        state=pushed; candidates=[pushed]
        for _ in range(8*self.level):
            choices=[self.transition(state,i) for i in range(4)]
            choices=[s for s in choices if s != state]
            if choices: state=self.rng.choice(choices)
            if self.projection(state)!=self.projection(self.state): candidates.append(state)
        self.target=tuple(self.projection(candidates[-1] if self.level>1 else pushed))

    def projection(self,state): return state[1][:self.relevant]

    def transition(self,state,command):
        p,boxes=state; n=self.n; dx,dy=DIRS[command]
        def ahead(q):
            x,y=q%n,q//n
            return (y+dy)*n+x+dx if 0<=x+dx<n and 0<=y+dy<n else -1
        q=ahead(p)
        if q<0: return state
        if q not in boxes: return (q,boxes)
        b=list(boxes); repeat=2 if self.impulse else 1
        for _ in range(repeat):
            chain=[]; cursor=q
            while cursor in b: chain.append(cursor); cursor=ahead(cursor)
            if cursor<0: break
            for cell in reversed(chain): b[b.index(cell)]=ahead(cell)
            if self.impulse: q=ahead(q)
        if tuple(b)==boxes: return state
        return (p if self.impulse else ahead(p),tuple(b))

    def draw(self,c):
        def point(p): return 9+(p%self.n)*6+self.offset[0],11+(p//self.n)*6+self.offset[1]
        c.rect(5,7,self.n*6+4,self.n*6+4,2,True)
        for i,p in enumerate(self.target): c.cross(*point(p),i,3)
        for i,p in enumerate(self.state[1]): c.token(*point(p),i,hollow=i>=self.relevant)
        x,y=point(self.state[0]); c.rect(x-1,y-1,3,3,0)



class Gravity(Puzzle):
    native_commands={3:0,4:1,5:2}
    def setup(self):
        self.n=3+self.level//2
        self.commands=('Step left then settle','Step right then settle','Reverse gravity')
        self.walls=frozenset((x,y) for x in range(1,self.n-1) for y in range(1,self.n-1)
                             if self.rng.random()<0.28)
        self.state=(0,self.n-1,1)
        self.reachable_target(2*self.level-1)

    def projection(self,state): return state[:2]

    def transition(self,state,command):
        x,y,g=state
        if command==2: g=-g
        else:
            xx=x+(-1 if command==0 else 1)
            if 0<=xx<self.n and (xx,y) not in self.walls: x=xx
        while 0<=y+g<self.n and (x,y+g) not in self.walls: y+=g
        return x,y,g

    def draw(self,c):
        ox,oy=10+self.offset[0],10+self.offset[1]
        c.rect(ox-4,oy-4,self.n*6+2,self.n*6+2,2,True)
        for x,y in self.walls: c.rect(ox+6*x-2,oy+6*y-2,5,5,2)
        c.cross(ox+6*self.target[0],oy+6*self.target[1],0,3)
        c.token(ox+6*self.state[0],oy+6*self.state[1],1)
        y=25; c.line((56,y-6),(56,y+6),0)
        c.line((53,y+3*self.state[2]),(56,y+6*self.state[2]),0)
        c.line((59,y+3*self.state[2]),(56,y+6*self.state[2]),0)


class Ghost(Puzzle):
    arrow_commands=(0,1,2,3)
    native_commands={5:5}
    def setup(self):
        self.d=1+(self.level-1)//2
        self.slots=1 if self.level<4 else 2
        self.exit_depth=self.level
        self.plates=((self.rng.choice((-1,1))*self.d,0),) if self.slots==1 else ((-self.d,0),(self.d,0))
        self.commands=('Up','Down','Left','Right','Record ghost / return','Wait')
        self.state=((0,0),(),(),0)
        self.target=(True,)*(self.slots+1)

    def move(self,p,command):
        if command>=4: return p
        dx,dy=DIRS[command]; q=(p[0]+dx,p[1]+dy)
        return q if (q[1]==0 and -self.d<=q[0]<=self.d) or (q[0]==0 and 0<=q[1]<=self.exit_depth) else p

    def ghost_positions(self,ghosts,tick):
        points=[]
        for path in ghosts:
            p=(0,0)
            for command in path[:tick]: p=self.move(p,command)
            points.append(p)
        return points

    def transition(self,state,command):
        p,record,ghosts,tick=state
        if command==4:
            if not record or len(ghosts)>=self.slots: return state
            return (0,0),(),ghosts+(record,),0
        points=self.ghost_positions(ghosts,tick+1)
        q=self.move(p,command)
        if q[1]>0 and not all(plate in points for plate in self.plates): q=p
        return q,record+(command,),ghosts,tick+1

    def subgoals(self,state=None):
        p,_,ghosts,tick=self.state if state is None else state
        points=self.ghost_positions(ghosts,tick)
        return [condition(f'plate-{i}',plate in points,True) for i,plate in enumerate(self.plates)] + [
            condition('exit',p,(0,self.exit_depth),depends_on=tuple(f'plate-{i}' for i in range(self.slots)))]

    def draw(self,c):
        def point(p): return 31+7*p[0],10+5*p[1]
        c.line(point((-self.d,0)),point((self.d,0)),2)
        c.line(point((0,0)),point((0,self.exit_depth)),2)
        for i,plate in enumerate(self.plates): c.token(*point(plate),i,True,size=7)
        c.cross(*point((0,self.exit_depth)),2,3)
        for p in self.ghost_positions(self.state[2],self.state[3]):
            x,y=point(p); c.rect(x-2,y-2,5,5,2,True)
        c.token(*point(self.state[0]),3)
        for i in range(self.slots): c.rect(51,15+8*i,5,5,0 if i<len(self.state[2]) else 2,True)

    def click_commands(self):
        x,y = self.state[0]
        half = self.size//2
        return [(4,(31+7*x-half,10+5*y-half,self.size,self.size))]


class Timing(Puzzle):
    native_commands={3:2,4:0,5:1}
    def setup(self):
        self.m=2*self.level-1
        self.period=2+min(3,self.level-1)
        self.phases={i:self.rng.randrange(self.period) for i in range(1,self.m,2)}
        self.commands=('Advance','Wait','Retreat')
        self.state=(0,0)
        self.target=(self.m,)
        self.budget=2*self.m*self.period+8

    def transition(self,state,command):
        p,t=state; t=(t+1)%self.period
        q=max(0,min(self.m,p+(1 if command==0 else -1 if command==2 else 0)))
        if q in self.phases and t==self.phases[q]: return None
        return q,t

    def projection(self,state): return (state[0],)

    def subgoals(self,state=None):
        p,_=self.state if state is None else state
        return [condition(f'crossing-{i}',p>i,True) for i in self.phases] + [condition('exit',p,self.m)]

    def draw(self,c):
        c.line((7,30),(7+4*self.m,30),2)
        for p,phase in self.phases.items():
            x=7+4*p; c.token(x,30,1,hollow=self.state[1]!=phase,size=3)
            c.line((x,24),(x,27),2)
            for t in range(self.period):
                c.rect(x-1,7+3*t,3,2,c.color(1) if t==phase else 2)
                if t==self.state[1]: c.rect(x,7+3*t,1,1,0)
        c.cross(7+4*self.m,30,0,2); c.token(7+4*self.state[0],30,2)
        # All hazard tracks align with this common clock. Downward progression
        # and the return edge expose order and wrap-around, without phase IDs.
        for t in range(self.period): c.rect(56,7+3*t,3,2,0 if t==self.state[1] else 2)
        last=7+3*(self.period-1)
        c.line((61,7),(61,last),2); c.arrow(61,last,0,1,2,1)
        c.line((61,last+2),(54,last+2),2)
        c.line((54,last+2),(54,5),2); c.line((54,5),(58,5),2)
        c.arrow(58,6,0,1,2,1)
