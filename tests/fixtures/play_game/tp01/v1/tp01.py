"""Locally authored protocol fixture, never evidence of official ARC gameplay.

A click marks the selected cell in blue and increments a visible one-pixel counter.
Space returns two frames (red, green); undo clears the mark. A click at (1, 1)
removes Space from the available actions, testing per-observation gating.
"""

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite


class Tp01(ARCBaseGame):
    def __init__(self, seed=0):
        super().__init__("tp01", levels=[Level(sprites=[
            Sprite([[9]], name="mark", x=32, y=32),
            Sprite([[0]], name="count", x=2, y=2),
        ], grid_size=(64, 64))], camera=Camera(background=5),
            available_actions=[5, 6, 7], seed=seed)

    def on_set_level(self, level):
        self.mark, self.counter = level.get_sprites()
        self.count = 0
        self.phase = 0
        self._available_actions = [5, 6, 7]

    def step(self):
        if self.action.id == GameAction.ACTION6:
            self.mark.set_position(self.action.data['x'], self.action.data['y'])
            self.count += 1
            self.counter.color_remap(None, self.count % 16)
            if self.action.data == {'x': 1, 'y': 1}:
                self._available_actions = [6, 7]
        elif self.action.id == GameAction.ACTION5:
            self.phase += 1
            self.mark.color_remap(None, 8 if self.phase == 1 else 14)
            if self.phase == 1:
                return
            self.phase = 0
        elif self.action.id == GameAction.ACTION7:
            self.mark.set_position(32, 32)
            self.mark.color_remap(None, 9)
        self.complete_action()
