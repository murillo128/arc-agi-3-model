"""One latent agent for training-owned modules, local evaluation and deployment.

The external Learner may own the very same Engine; inference never creates an
optimizer, writes replay or learns from imagined transitions. A selected action
is pending until its real result arrives, including SDK RESET.
"""
from dataclasses import dataclass, field
import random

import numpy as np
import torch

from .data import Action, Observation, random_action
from .learning import State
from .planning import PlanConfig, Planner
from .goal_planning import GoalConfig, GoalPlanner


@dataclass(frozen=True)
class AgentConfig:
    mode: str = 'value_mpc'
    max_actions: int = 128
    planner: PlanConfig = field(default_factory=PlanConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)

    def __post_init__(self):
        if self.mode not in {'random', 'q', 'mpc', 'value_mpc', 'goal_mpc', 'goal_value_mpc'}:
            raise ValueError('Unknown agent policy mode')
        if type(self.max_actions) is not int or self.max_actions < 1:
            raise ValueError('max_actions must be a positive integer')


def public_observation(raw, action=None):
    """Accept the shared public snapshot or an SDK object, never private rules."""
    if isinstance(raw, Observation):
        return raw
    if not isinstance(getattr(raw, 'state', None), str):
        return Observation.from_sdk(raw, action)
    frames = np.asarray(raw.frame)
    if frames.size == 0:
        frames = np.empty((0, 0, 0), dtype=np.uint8)
    elif frames.ndim != 3 or min(frames.shape) < 1 or np.any((frames < 0) | (frames > 15)):
        raise ValueError('Expected native rectangular ARC palette frames [F,H,W]')
    frames = np.array(frames, dtype=np.uint8, copy=True)
    frames.flags.writeable = False
    return Observation(frames, tuple(map(int, raw.available_actions)), raw.state,
                       int(raw.levels_completed), int(raw.win_levels), action)


class Agent:
    def __init__(self, engine, heads, config=AgentConfig(), *, seed=0,
                 frozen=True, compile_heads=False, proposal_policy=None):
        self.engine, self.heads, self.config = engine, heads, config
        self.proposal_policy = proposal_policy
        # Planner construction freezes parameters for inference. An externally
        # trained agent explicitly preserves the existing optimizer ownership.
        parameters = list(engine.model.parameters()) + list(heads.parameters())
        flags = [p.requires_grad for p in parameters]
        cls = GoalPlanner if config.mode in {'goal_mpc', 'goal_value_mpc'} else Planner
        planner_config = config.goal if cls is GoalPlanner else config.planner
        self.planner = cls(engine, heads, planner_config, compile_heads=compile_heads)
        self.planner.proposal_policy = proposal_policy
        if frozen:
            for module in (engine.model, heads, proposal_policy):
                if module is not None:
                    module.eval().requires_grad_(False)
        else:
            for parameter, flag in zip(parameters, flags):
                parameter.requires_grad_(flag)
        self.seed = seed
        self.reset(seed=seed)

    @property
    def remaining_budget(self):
        return max(0, self.config.max_actions - self.actions_taken)

    def reset(self, *, seed=None):
        """Start a new game lifetime. SDK RESET itself does not replenish budget."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.state = None
        self.observation = None
        self.pending_action = None
        self.actions_taken = 0
        self.last_diagnostics = {}
        if hasattr(self.planner, 'reset'):
            self.planner.reset()

    @torch.no_grad()
    def observe(self, observation, action=None):
        """Assimilate exactly one confirmed response; callers own event ordering."""
        observation = public_observation(observation, action or self.pending_action)
        executed = action or observation.action_input or self.pending_action
        before = self.observation
        if self.state is None or before is None:
            self.state = self.engine.initial(observation)
        else:
            if executed is None:
                raise ValueError('A subsequent observation requires its executed action')
            if self.pending_action is not None and executed != self.pending_action:
                raise ValueError('Confirmed action does not match the pending action')
            if hasattr(self.planner, 'after_real_action'):
                self.planner.after_real_action(executed, before, observation)
            # Old-level animation frames are not part of a new-level state.
            if executed.id == 0 or observation.starts_new_level(before):
                self.state = self.engine.initial(observation.level_initial())
                if hasattr(self.planner, 'reset'):
                    self.planner.reset()
            else:
                self.state = self.engine.observe(self.state, executed, observation)
            self.actions_taken += 1
        self.observation, self.pending_action = observation, None
        return self.state

    @torch.no_grad()
    def act(self, *, diagnostics=False):
        """Choose once; selection and imaginary branches do not spend real budget."""
        if self.observation is None:
            raise ValueError('Observe the initial public frame before acting')
        if self.pending_action is not None:
            raise ValueError('Await the real result of the pending action')
        if not self.remaining_budget:
            raise ValueError('Action budget exhausted')
        obs = self.observation
        self.last_diagnostics = {}
        if obs.state == 'WIN':
            raise ValueError('Cannot act after WIN')
        if obs.state in {'NOT_PLAYED', 'GAME_OVER'}:
            chosen = Action(0)
        elif self.config.mode == 'random' or self.state.z is None:
            chosen = random_action(obs, self.rng)
        else:
            chosen = self.planner.choose(
                self.state, obs, self.rng, self.config.mode,
                diagnostics=self.last_diagnostics if diagnostics else None,
                remaining_budget=self.remaining_budget)
        self.pending_action = chosen
        return chosen

    def choose_action(self, observation):
        """Shared Policy adapter used by both local and official SDK loops."""
        from arc3.core.policy import Action as PublicAction
        self.observe(observation)
        action = self.act()
        return PublicAction(action.id, dict(action.data))

    @torch.no_grad()
    def imagine(self, actions, *, state=None):
        """Independent latent rollout; never changes real state, RNG or budget."""
        origin = state if state is not None else self.state
        if origin is None or origin.z is None or origin.geometry is None:
            raise ValueError('Imagination requires a confirmed visual state')
        z, h = origin.z[None].clone(), origin.h.clone()
        result = []
        self.engine.model.eval()
        with self.engine.autocast():
            for action in actions:
                z, h, _ = self.engine.predict(z, h, self.engine.actions([action]), *origin.geometry)
                result.append(State(z[0].detach().clone(), h.detach().clone(), origin.geometry))
        return result

    def trace_diagnostics(self, observation, action):
        """Read-only snapshot: tracing must never assimilate a frame twice."""
        import copy
        return copy.deepcopy(self.last_diagnostics)
