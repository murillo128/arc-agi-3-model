"""Original synthetic mechanics; no SDK or evaluated-policy dependencies."""
from __future__ import annotations

from games.skill_common import Canvas, DIRS, Puzzle, condition


class Dials(Puzzle):
    def setup(self):
        self.n = self.level
        self.mod = 2 if self.level <= 2 else min(4,2+(self.level-1)//2)
        self.coupled = self.spec.skill == 'efectos_acoplados'
        self.relational = self.spec.skill in ('igualdad_desigualdad','restricciones_locales_globales')
        self.commands = tuple(f'Cycle dial {i+1}' for i in range(self.n))
        self.state = (0,)*self.n
        self.reachable_target(max(self.n,2*self.level-1))
        if not self.coupled:
            self.target = tuple(self.rng.randrange(1,self.mod) for _ in range(self.n))
        self.equal = tuple(self.target[i] == self.target[i+1] for i in range(self.n-1))

    def transition(self,state,command):
        s = list(state); s[command] = (s[command]+1)%self.mod
        if self.coupled and command+1 < self.n: s[command+1]=(s[command+1]+1)%self.mod
        return tuple(s)

    def subgoals(self,state=None):
        s = self.state if state is None else state
        if not self.relational: return super().subgoals(s)
        return [condition('anchor',s[0],self.target[0])] + [
            condition(f'link-{i}',s[i] == s[i+1],equal) for i,equal in enumerate(self.equal)]

    def draw(self,c):
        if not self.relational:
            super().draw(c)
        else:
            c.token(7,15,self.target[0],True,size=7)
            for i,value in enumerate(self.state):
                c.token(7+9*i,31,value)
                if i < self.n-1:
                    c.rect(11+9*i,20,3,1,0)
                    if self.equal[i]: c.rect(11+9*i,22,3,1,0)
                    else: c.line((11+9*i,23),(13+9*i,19),0)
        for v in range(self.mod): c.token(5+6*v,45,v,size=3)
        if self.coupled:
            for i in range(self.n-1): c.line((6+7*i,34),(13+7*i,34),2)



class Keys(Puzzle):
    def setup(self):
        self.n = min(3,self.level)
        self.mod = 2 if self.level < 3 else self.level+1
        self.gated = self.spec.skill == 'precondiciones'
        self.commands = tuple(f'Change attribute {i+1}' for i in range(self.n))
        if self.spec.skill == 'composicion_de_transformaciones' and self.level >= 3:
            self.commands += ('Swap first and last attribute',)
        self.state = (0,)*self.n
        self.target=tuple(self.rng.randrange(1,self.mod) for _ in range(self.n))

    def transition(self,state,command):
        s=list(state)
        if command == self.n: s[0],s[-1]=s[-1],s[0]
        elif not self.gated or command == 0 or s[command-1] == self.mod-1:
            s[command]=(s[command]+1)%self.mod
        return tuple(s)

    def draw(self,c):
        super().draw(c)
        for i in range(self.n):
            c.digit(5+7*i,36,i+1)
            if self.gated and i:
                c.token(6+7*i,44,self.mod-1,True,size=5)
        c.rect(1,10,7*self.n+6,22,2,True)



class Paint(Puzzle):
    def setup(self):
        self.n = 2 + (self.level-1)//2
        n=self.n
        self.masks=[tuple(range(n)),tuple(range((n-1)*n,n*n))]
        if self.level >= 2: self.masks += [tuple(range(0,n*n,n)),tuple(range(n-1,n*n,n))]
        if self.level >= 4: self.masks += [tuple(i*n+i for i in range(n))]
        self.commands=tuple(f'Paint region {i//2+1} with color {i%2+1}' for i in range(len(self.masks)*2))
        self.state=(0,)*(n*n)
        self.reachable_target(2*self.level-1)

    def transition(self,state,command):
        s=list(state)
        for i in self.masks[command//2]: s[i]=command%2+1
        return tuple(s)

    def subgoals(self,state=None):
        s=self.state if state is None else state
        return [condition(f'row-{r}',list(s[r*self.n:(r+1)*self.n]),list(self.target[r*self.n:(r+1)*self.n])) for r in range(self.n)]

    def draw(self,c):
        n=self.n; unit=self.size+1
        for k,s in enumerate((self.target,self.state)):
            for i,v in enumerate(s): c.rect(4+30*k+(i%n)*unit,13+(i//n)*unit,unit-1,unit-1,c.color(v))
            c.rect(2+30*k,10,unit*n+2,unit*n+5,0 if k==0 else 2,True)
        for i,mask in enumerate(self.masks):
            c.rect(4+11*i,42,8,5,2,True)
            for j in mask: c.rect(5+11*i+j%n,43+j//n,1,1,0)



class Loops(Puzzle):
    def setup(self):
        self.n=min(8,1+self.level)
        self.cycles=(tuple(range(self.n)),tuple(range(min(3,self.n))))
        self.commands=('Loop A forward','Loop A backward','Loop B forward','Loop B backward')
        self.state=tuple(range(self.n))
        self.reachable_target(2*self.level-1)

    def transition(self,state,command):
        s=list(state); cycle=self.cycles[command//2]; step=1 if command%2==0 else -1
        for i,p in enumerate(cycle): s[cycle[(i+step)%len(cycle)]]=state[p]
        return tuple(s)

    def draw(self,c):
        super().draw(c)
        c.line((6,36),(6+7*(self.n-1),36),c.color(8))
        c.line((6,40),(6+7*(len(self.cycles[1])-1),40),c.color(9))
        c.rect(4,32,3,9,2,True)



class Pegs(Puzzle):
    board_controls=True
    def setup(self):
        self.n=3+self.level//2
        n=self.n; board=[0]*(n*n); board[self.rng.choice((0,n-1,n*(n-1),n*n-1))]=1
        used={i for i,v in enumerate(board) if v}
        for _ in range(self.level):
            options=[]
            for p,v in enumerate(board):
                if v!=1: continue
                x,y=p%n,p//n
                for dx,dy in DIRS:
                    if 0<=x+2*dx<n and 0<=y+2*dy<n:
                        q=(y+dy)*n+x+dx; r=(y+2*dy)*n+x+2*dx
                        if board[q]==board[r]==0: options.append((p,q,r))
            if not options: break
            p,q,r=self.rng.choice(options); board[p]=0; board[q]=board[r]=1; used.update((p,q,r))
        if self.spec.skill=='reescritura_condicional' and self.level>=3:
            for p in self.rng.sample([p for p in range(n*n) if p not in used], min(self.level-2,n*n-len(used))): board[p]=2
        self.state=(tuple(board),-1)
        self.commands=tuple(f'Pick peg or landing {i+1}' for i in range(n*n))
        self.target=(1,)

    def transition(self,state,command):
        board,selected=state
        if board[command]==1: return (board,command)
        if selected<0 or board[command]: return state
        x,y=selected%self.n,selected//self.n; u,v=command%self.n,command//self.n
        if (abs(x-u),abs(y-v)) not in ((2,0),(0,2)): return state
        mid=((y+v)//2)*self.n+(x+u)//2
        if not board[mid]: return state
        b=list(board); b[selected]=0; b[command]=1
        if b[mid]==1: b[mid]=0
        return (tuple(b),-1)

    def projection(self,state): return (state[0].count(1),)

    def button_rects(self): return [(6+(i%self.n)*7,8+(i//self.n)*7,6,6) for i in range(self.n*self.n)]

    def draw(self,c):
        for i,(x,y,w,h) in enumerate(self.button_rects()):
            v=self.state[0][i]; c.rect(x,y,w,h,2,True)
            if v: c.token(x+w//2,y+h//2,v-1)
            if i==self.state[1]: c.rect(x,y,w,h,0,True)
        c.token(54,15,0,size=3); c.digit(53,23,1)



class Agents(Puzzle):
    def setup(self):
        self.m=2+self.level
        self.role=self.rng.choice((-1,1)) if self.spec.skill=='inferencia_de_roles' else 1
        self.commands=('Move left','Move right','Wait')
        self.state=(0,0)
        if self.level==1: self.reachable_target(1)
        else: self.target=(self.m-self.rng.randrange(2),self.rng.randrange(self.m+1))

    def transition(self,state,command):
        p,q=state
        p=max(0,min(self.m,p+(-1 if command==0 else 1 if command==1 else 0)))
        return p,(q+self.role)%(self.m+1)

    def draw(self,c):
        for i in range(2):
            y=19+16*i; c.line((6,y),(6+6*self.m,y),2)
            c.cross(6+6*self.target[i],y,i,3)
            c.token(6+6*self.state[i],y,i)
        c.rect(54,32,5,5,2,True)  # The second mover is autonomous, not selected.



class Tanks(Puzzle):
    def setup(self):
        self.n=min(4,2+(self.level-1)//2)
        self.caps=tuple(self.rng.randint(2+self.level//2,3+self.level) for _ in range(self.n))
        self.commands=tuple(f'Transfer {i//2+1} {"right" if i%2==0 else "left"}' for i in range(2*(self.n-1)))
        self.state=(self.caps[0],)+(0,)*(self.n-1)
        self.reachable_target(2*self.level-1)

    def transition(self,state,command):
        a,b=command//2,command//2+1
        if command%2: a,b=b,a
        if state[a]==0 or state[b]==self.caps[b]: return state
        s=list(state); s[a]-=1; s[b]+=1; return tuple(s)

    def draw(self,c):
        for i,cap in enumerate(self.caps):
            x=5+14*i; bottom=44; unit=self.size-1; width=7+self.size
            c.rect(x,bottom-unit*cap,width,unit*cap+1,2,True)
            c.rect(x+1,bottom-unit*self.state[i],width-2,unit*self.state[i],c.color(0))
            c.line((x-1,bottom-unit*self.target[i]),(x+width,bottom-unit*self.target[i]),c.color(1))



class Flow(Puzzle):
    def setup(self):
        self.n=self.level
        self.split=self.spec.skill=='bifurcacion'
        self.commands=tuple(f'Toggle valve {i+1}' for i in range(self.n))+('Send signal',)
        self.run_command=self.n
        bits=tuple(self.rng.randrange(2) for _ in range(self.n))
        self.state=(bits,False)
        # Each stage affects distinct sources. Random initial and target routes
        # avoid a universal "turn every valve on" solution across seeds.
        changed=set(self.rng.sample(range(self.n),1+self.level//2 if self.level>1 else 1))
        wanted=tuple(1-bit if i in changed else bit for i,bit in enumerate(bits))
        self.target=self.output(wanted)

    def transition_frames(self,before,command,after):
        if command!=self.run_command: return []
        from copy import copy
        frames=[]
        for stage in range(1,self.n+1):
            view=copy(self); view.state=after; view._render_stage=stage
            frames.append(view.render())
        return frames

    def output(self,bits):
        if hasattr(self,'_render_stage'): bits=bits[:self._render_stage]
        active=[1<<i for i in range(self.n+1)]
        for i,bit in enumerate(bits):
            if bit:
                if self.split: active[i+1] |= active[i]
                else: active[i],active[i+1]=active[i+1],active[i]
        return tuple(active)

    def projection(self,state): return self.output(state[0])

    def transition(self,state,command):
        bits,_=state
        if command==self.n: return bits,True
        b=list(bits); b[command]=1-b[command]; return tuple(b),False

    def subgoals(self,state=None):
        s=self.state if state is None else state
        return [condition(f'outlet-{i}',v,self.target[i]) for i,v in enumerate(self.output(s[0]))] + [condition('tested',s[1],True)]

    def draw(self,c):
        for lane in range(self.n+1):
            y=10+5*lane; c.line((5,y),(52,y),2)
            c.token(3,y,lane,size=3)
            for bit in range(self.n+1):
                if self.target[lane]&(1<<bit): c.rect(55+bit,y-2,1,2,c.color(bit))
                if self.state[1] and self.output(self.state[0])[lane]&(1<<bit):
                    c.rect(55+bit,y+1,1,2,c.color(bit))
        for i,bit in enumerate(self.state[0]):
            x=9+7*i; y=10+5*i
            c.line((x,y),(x+4,y+5),c.color(i) if bit else 2)
            c.token(x+2,y+2,i,hollow=not bit,size=self.size)



class Merge(Puzzle):
    board_controls=True
    def setup(self):
        self.n=2*self.level
        self.commands=tuple(f'Pick or merge block {i+1}' for i in range(self.n))
        self.state=((0,)*self.n,-1)
        b=list(self.state[0])
        for _ in range(max(1,self.level)):
            pairs=[(i,j) for i in range(self.n) for j in range(i+1,self.n) if b[i]>=0 and b[i]==b[j]]
            if not pairs: break
            i,j=self.rng.choice(pairs); b[j]+=1; b[i]=-1
        self.target=tuple(b.count(t) for t in range(5))

    def transition(self,state,command):
        b,selected=state
        if b[command]<0: return state
        if selected<0 or selected==command: return b,command if selected<0 else -1
        if b[selected]!=b[command]: return b,command
        result=list(b); result[command]+=1; result[selected]=-1
        return tuple(result),-1

    def projection(self,state): return tuple(state[0].count(t) for t in range(5))

    def button_rects(self): return [(5+(i%6)*9,23+(i//6)*13,8,9) for i in range(self.n)]

    def draw(self,c):
        for t,count in enumerate(self.target):
            x=7+11*t; c.token(x,10,t,True,size=3+t)
            c.digit(x-1,15,count)
        for i,(x,y,w,h) in enumerate(self.button_rects()):
            v=self.state[0][i]
            if v>=0: c.token(x+4,y+4,v,size=min(7,self.size+v))
            if i==self.state[1]: c.rect(x,y,w,h,0,True)



