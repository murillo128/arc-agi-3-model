"""Seedable micro-skill catalog and public factory (no SDK dependency).

Related skill profiles share explicit mechanical contexts, not fixed solutions.
Privileged labels are returned only through the opt-in teacher methods.
"""
from __future__ import annotations

from games.skill_common import Canvas, LEVELS, Puzzle, SkillSpec
from games.skill_spatial import (Boxes, Centroid, Ghost, Gravity, Linkage, Maze,
                                Timing, Transform, Twins)
from games.skill_systems import (Agents, Dials, Flow, Keys, Loops, Merge, Paint,
                                Pegs, Tanks)
from games.skill_symbolic import Memory, Program, Rules, Sigils, execute_program


# Stable IDs. Contexts share mechanics, not fixed scenes or solutions.
_ROWS = (
    ('igualdad_desigualdad', 'Equality and inequality', 'dials'),
    ('conjuncion_de_restricciones', 'All constraints at once', 'dials'),
    ('restricciones_locales_globales', 'Linked constraints', 'dials'),
    ('estados_ciclicos', 'Cyclic states', 'dials'),
    ('efectos_acoplados', 'Coupled effects', 'dials'),
    ('estado_factorizado', 'Factorized keys', 'keys'),
    ('composicion_de_transformaciones', 'Transform a key', 'keys'),
    ('precondiciones', 'Unlock before changing', 'keys'),
    ('planificacion_en_estado_ampliado', 'Position and key state', 'maze'),
    ('correspondencia_relacional', 'Relational matching', 'rules'),
    ('transformaciones_geometricas', 'Geometric transforms', 'transform'),
    ('simetria', 'Mirror images', 'transform'),
    ('marcos_de_referencia', 'Moving reference frame', 'transform'),
    ('cobertura_de_conjuntos', 'Cover every mark', 'transform'),
    ('razonamiento_inverso', 'Infer a preimage', 'transform'),
    ('occlusion_y_capas', 'Paint layers', 'paint'),
    ('operaciones_sobre_regiones', 'Paint regions', 'paint'),
    ('orden_no_conmutativo', 'Order of painting', 'paint'),
    ('invariantes', 'Mirrored invariants', 'twins'),
    ('ruptura_de_invariantes', 'Break the symmetry', 'twins'),
    ('obstaculos_como_herramientas', 'Use a wall as a tool', 'twins'),
    ('centroide', 'Move the centroid', 'centroid'),
    ('relaciones_parte_todo', 'Parts move the whole', 'centroid'),
    ('control_indirecto', 'Move through another object', 'boxes'),
    ('dependencias_jerarquicas', 'Parent and child rods', 'linkage'),
    ('subobjetivos_no_monotonos', 'Retract before extending', 'linkage'),
    ('topologia_dinamica', 'Switchable bridges', 'maze'),
    ('invariantes_de_seguridad', 'Keep a safe foothold', 'maze'),
    ('navegacion_en_grafos', 'Connected rooms', 'maze'),
    ('recursos_limitados', 'Route on a budget', 'maze'),
    ('permutaciones', 'Overlapping loops', 'loops'),
    ('secuencias_ordenadas', 'Order the tokens', 'loops'),
    ('evitar_estados_sin_salida', 'Do not strand a peg', 'pegs'),
    ('reescritura_condicional', 'Conditional capture', 'pegs'),
    ('dinamica_por_contacto', 'Contact and impulse', 'boxes'),
    ('relevancia_del_objetivo', 'Deliver only what counts', 'boxes'),
    ('coordinacion_multiagente', 'Coordinate two movers', 'agents'),
    ('inferencia_de_roles', 'Discover the other mover', 'agents'),
    ('sincronizacion', 'Arrive together', 'agents'),
    ('conservacion_y_transferencia', 'Transfer, do not create', 'tanks'),
    ('limites_de_capacidad', 'Respect tank capacity', 'tanks'),
    ('propagacion', 'Route a signal', 'flow'),
    ('bifurcacion', 'Control every branch', 'flow'),
    ('simulacion_causal', 'Predict the entire flow', 'flow'),
    ('dinamica_autonoma', 'Move then settle', 'gravity'),
    ('cardinalidad_exacta', 'Merge to exact counts', 'merge'),
    ('composicion_de_funciones', 'Compose dictionaries', 'rules'),
    ('invariancia_perceptiva', 'Symbols across appearances', 'rules'),
    ('simbolos_y_significado', 'Draw an operator', 'sigils'),
    ('codificacion_binaria', 'Binary instructions', 'program'),
    ('sintesis_de_programas', 'Build a program', 'program'),
    ('subrutinas', 'Reusable calls', 'program'),
    ('pila_de_ejecucion', 'Return to the caller', 'program'),
    ('recursion', 'Nested recursive calls', 'program'),
    ('memoria_parcial', 'Remember hidden pairs', 'memory'),
    ('tipos_latentes', 'Probe hidden types', 'memory'),
    ('exploracion_informativa', 'Reveal before choosing', 'memory'),
    ('historial_como_estado', 'Record a ghost helper', 'ghost'),
    ('fase_temporal', 'Cross in the right phase', 'timing'),
    ('orden_de_eventos', 'Move before the hazard', 'timing'),
)
SPECS = tuple(SkillSpec(f'sy{i:02d}', *row) for i, row in enumerate(_ROWS, 1))
CATALOG = {s.skill: s for s in SPECS}
BY_ID = {s.game_id: s for s in SPECS}


CONTEXTS = dict(dials=Dials, keys=Keys, transform=Transform, paint=Paint,
                twins=Twins, centroid=Centroid, linkage=Linkage, maze=Maze,
                loops=Loops, pegs=Pegs, boxes=Boxes, agents=Agents, tanks=Tanks,
                flow=Flow, gravity=Gravity, merge=Merge, rules=Rules,
                program=Program, sigils=Sigils, memory=Memory, ghost=Ghost,
                timing=Timing)


def make_puzzle(skill: str, seed: int=0, level: int=1, *, appearance_seed: int | None=None) -> Puzzle:
    if not isinstance(skill,str): raise ValueError('skill must be a string')
    if appearance_seed is not None and type(appearance_seed) is not int: raise ValueError('appearance_seed must be an integer')
    spec=CATALOG.get(skill) or BY_ID.get(skill.removesuffix('-v1'))
    if spec is None: raise ValueError(f'Unknown synthetic skill: {skill!r}')
    return CONTEXTS[spec.context](spec,seed,level,appearance_seed=appearance_seed)
