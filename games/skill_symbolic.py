"""Original synthetic mechanics; no SDK or evaluated-policy dependencies."""
from __future__ import annotations

from games.skill_common import Canvas, DIRS, Puzzle, condition


def glyph(c, x, y, value, color, rotation=0):
    scale=1 if c.size==3 else 2
    shapes=(((1,1),),((0,0),(0,1),(0,2),(1,2)),
            ((1,0),(0,1),(1,1),(2,1),(1,2)),
            ((0,0),(1,0),(2,0),(0,1),(2,1),(0,2),(1,2),(2,2)))
    for u,v in shapes[value%4]:
        for _ in range(rotation%4): u,v=2-v,u
        c.rect(x+u*scale,y+v*scale,scale,scale,color)



class Rules(Puzzle):
    board_controls=True
    def setup(self):
        self.n=self.level
        self.k=min(4,1+self.level)
        self.depth=(1+(self.level-1)//2) if self.spec.skill=='composicion_de_funciones' else 1
        self.maps=tuple(tuple(self.rng.sample(range(self.k),self.k)) for _ in range(self.depth))
        self.phrase=tuple(self.rng.randrange(self.k) for _ in range(self.n))
        answer=self.phrase
        for table in self.maps: answer=tuple(table[v] for v in answer)
        self.target=answer
        self.state=tuple((v+1)%self.k for v in answer)
        self.commands=tuple(f'Cycle answer glyph {i+1}' for i in range(self.n))
        self.rotations=tuple(self.rng.randrange(4) for _ in range(40)) if self.spec.skill=='invariancia_perceptiva' else (0,)*40

    def transition(self,state,command):
        s=list(state); s[command]=(s[command]+1)%self.k; return tuple(s)

    def button_rects(self): return [(6+9*i,38,8,9) for i in range(self.n)]

    def draw(self,c):
        for depth,table in enumerate(self.maps):
            for i,v in enumerate(table):
                x=3+15*i; y=8+8*depth
                glyph(c,x,y,i,c.color(depth),self.rotations[depth*8+i])
                c.line((x+6,y+1),(x+7,y+1),2)
                glyph(c,x+9,y,v,c.color(depth+1),self.rotations[depth*8+4+i])
        for i,v in enumerate(self.phrase): glyph(c,8+9*i,32,v,c.color(0),self.rotations[25+i])
        for i,(x,y,w,h) in enumerate(self.button_rects()):
            c.rect(x,y,w,h,2,True)
            glyph(c,x+1,y+1,self.state[i],c.color(self.depth),self.rotations[32+i])



def execute_program(codes: tuple[int,...], *, call_depth=1, recursive_arguments=False):
    """A bounded stack interpreter. Calls really suspend and resume their caller.

    Instructions 1 and 2 emit two symbols; 0 is empty. Instruction 3 calls F.
    F(d) = emit(0); F(d-1) when d>1; emit(1). In argument mode a main slot
    contains the depth passed to F (0 means no call). No Python eval/exec.
    """
    stack=[('main',codes,0)]; output=[]; peak=1; count=0
    while stack:
        kind,body,pc=stack.pop()
        if pc==len(body): continue
        stack.append((kind,body,pc+1)); op=body[pc]; count+=1
        if count>256: raise ValueError('program exceeds bounded interpreter limit')
        if kind=='main' and recursive_arguments:
            if op: stack.append(('function',(1,('call',op-1),2),0))
        elif isinstance(op,tuple):
            if op[1]: stack.append(('function',(1,('call',op[1]-1),2),0))
        elif op==3: stack.append(('function',(1,('call',call_depth-1),2),0))
        elif op in (1,2): output.append(op-1)
        peak=max(peak,len(stack))
    return tuple(output),peak



class Program(Puzzle):
    def setup(self):
        self.recursive=self.spec.skill=='recursion'
        self.n=min(3,1+(self.level-1)//2) if self.recursive else min(6,self.level)
        self.depth=1+(self.level-1)//2 if self.spec.skill=='pila_de_ejecucion' else 1
        self.commands=tuple(f'Toggle bit {i%2} of instruction {i//2+1}' for i in range(2*self.n))+('Run program',)
        self.run_command=2*self.n
        self.state=((0,)*self.n,(),False)
        values=tuple(self.rng.randint(1,min(3,self.level)) for _ in range(self.n))
        if self.spec.skill in ('subrutinas','pila_de_ejecucion'): values=(3,)+values[1:]
        self.target=execute_program(values,call_depth=self.depth,recursive_arguments=self.recursive)[0]

    def transition(self,state,command):
        codes,_,_=state
        if command==self.run_command:
            out,_=execute_program(codes,call_depth=self.depth,recursive_arguments=self.recursive)
            return codes,out,True
        codes=list(codes); codes[command//2]^=1<<(command%2)
        return tuple(codes),(),False

    def transition_frames(self,before,command,after):
        if command!=self.run_command: return []
        codes,out,_=after
        return [self.render_state((codes,out[:i],False)) for i in range(1,len(out)+1)]

    def subgoals(self,state=None):
        _,out,ran=self.state if state is None else state
        return [condition(f'output-{i}',out[i] if i<len(out) else None,v) for i,v in enumerate(self.target)] + [
            condition('output-length',len(out),len(self.target)),condition('executed',ran,True)]

    def draw(self,c):
        for row,seq in enumerate((self.target,self.state[1])):
            for i,v in enumerate(seq[:36]): c.token(4+3*(i%18),9+row*10+(i//18)*4,v,size=max(2,self.size-2))
        for i,code in enumerate(self.state[0]):
            x=5+9*i; c.rect(x,30,7,8,2,True)
            for b in range(2): c.rect(x+2+2*b,32,1,3,0 if code&(1<<b) else 2)
        demo=execute_program((3,),call_depth=self.depth,recursive_arguments=self.recursive)[0]
        c.digit(3,41,3)
        for i,v in enumerate(demo): c.token(12+3*i,43,v,size=2)



class Sigils(Puzzle):
    board_controls=True
    def setup(self):
        from itertools import combinations
        self.n=min(3,self.level)
        self.mod=2 if self.level<3 else 3
        weight=min(4,self.level)
        options=list(combinations(range(9),weight))
        self.patterns=tuple(sum(1<<i for i in cells) for cells in self.rng.sample(options,self.n))
        self.state=((0,)*self.n,0)
        self.target=tuple(self.rng.randrange(1,self.mod) for _ in range(self.n))
        self.commands=tuple(f'Toggle sigil cell {i+1}' for i in range(9))

    def transition(self,state,command):
        attrs,mask=state; mask^=1<<command
        if mask in self.patterns:
            s=list(attrs); i=self.patterns.index(mask); s[i]=(s[i]+1)%self.mod
            return tuple(s),0
        return attrs,mask

    def projection(self,state): return state[0]

    def button_rects(self): return [(20+(i%3)*8,24+(i//3)*8,6,6) for i in range(9)]

    def draw(self,c):
        for i,mask in enumerate(self.patterns):
            for b in range(9):
                if mask&(1<<b): c.rect(5+18*i+(b%3)*2,8+(b//3)*2,1,1,c.color(i))
            c.digit(13+18*i,8,self.target[i],c.color(i))
            c.digit(13+18*i,16,self.state[0][i],c.color(i))
        for i,(x,y,w,h) in enumerate(self.button_rects()):
            c.rect(x,y,w,h,2,True)
            c.rect(x+2,y+2,self.size-2,self.size-2,0 if self.state[1]&(1<<i) else 2)



class Memory(Puzzle):
    board_controls=True
    def setup(self):
        self.n=self.level
        deck=list(range(self.n))*2; self.rng.shuffle(deck); self.deck=tuple(deck)
        self.state=(0,-1,-1)
        self.commands=tuple(f'Reveal card {i+1}' for i in range(2*self.n))
        self.target=(True,)*self.n

    def transition(self,state,command):
        removed,opened,_=state
        if removed&(1<<command) or opened==command: return state
        if opened<0: return removed,command,-1
        if self.deck[opened]==self.deck[command]:
            return removed|(1<<opened)|(1<<command),-1,-1
        return removed,-1,-1

    def transition_frames(self,before,command,after):
        if before[1]>=0 and before[1]!=command and not before[0]&(1<<command):
            return [self.render_state((before[0],before[1],command))]
        return []

    def subgoals(self,state=None):
        removed,_,_=self.state if state is None else state
        return [condition(f'pair-{i}',all(removed&(1<<j) for j,v in enumerate(self.deck) if v==i),True,monotonic=True) for i in range(self.n)]

    def button_rects(self): return [(5+(i%6)*9,12+(i//6)*17,5+self.size//2,8+self.size) for i in range(2*self.n)]

    def draw(self,c):
        for i,(x,y,w,h) in enumerate(self.button_rects()):
            if self.state[0]&(1<<i): c.rect(x+3,y+5,1,1,2); continue
            c.rect(x,y,w,h,2,True)
            if i in self.state[1:]:
                c.token(x+3,y+5,self.deck[i])
            else: c.rect(x+2,y+3,3,5,c.color(9),True)



