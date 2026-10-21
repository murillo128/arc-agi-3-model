"""Human control documentation. Never imported by an agent or SDK adapter."""
from games.skill_curriculum import SkillGame
from games.skill_core import LATER_PRACTICE
from scripts.skill_catalog import ACTION_LABELS, control_descriptions

CONTEXT_HELP = {
    'basic_color': 'Tap each filled tile to toggle between two colours. Match its nearby hollow target. Tiles are independent.',
    'basic_unlock': 'Tap an enabled switch to open it permanently. Follow the visible connections: a later switch opens only after every incoming switch is open. Open all switches and their connected goals. Undo can close a switch again.',
    'basic_move': 'Use arrows to move the solid object onto its matching hollow target. Each arrow moves one grid step; the object is already selected.',
    'basic_rotate': 'Tap a piece to turn it clockwise around its fixed white centre. Fit the filled cells into the hollow outline. Each piece turns independently.',
    'basic_reflect': 'Move the solid object with arrows. Its outlined reflection moves across the fixed vertical mirror. Bring the reflection onto the hollow target on the other side. Horizontal reflection reverses left and right.',
    'basic_loop': 'Match the left loop to the hollow reference on the right. Right turns all tokens one position clockwise; Left turns them counterclockwise.',
    'basic_transfer': 'Match the liquid in each tank to the small hollow target slots beside it. Left/Right transfer on the first connection; Up/Down transfer on the second when present. Empty sources and full destinations block a transfer; liquid is conserved.',
    'basic_pair': 'Tap two cards to reveal their colours. Equal colours stay revealed with a matched underline. A mismatch shows both faces briefly, then hides them. Find every pair. Cards never move during a level.',

    'dials': 'Tap a lower dial to cycle it. Hollow upper tokens are goals. Paired samples on a link require the neighbouring dials to be alike or different; only the first has an exact anchor in relational games. Outgoing arrows show coupled effects. Bottom samples and the return line show cycle order.',
    'keys': 'Tap each boxed attribute to match its hollow upper goal. Bottom samples show the ordered cycle and return. In gated games the sample on the lower connection is the required predecessor state, separate from the upper goal. The crossbar opens when enabled. Space swaps the first and last attributes when available.',
    'transform': 'Cover every target cross with the shape. Arrow actions move it; Space rotates it clockwise. Tap a filled cell of the figure to reflect it along its local axis, which turns with the piece. Mirror profiles show a reflected copy. In moving-reference profiles Space cycles through legal mirror positions instead of rotating.',
    'paint': 'Reproduce the left reference on the right canvas. Up/Down paint the top/bottom row; Left/Right paint the left/right column; Space paints the diagonal when available. Each region has one fixed colour. Overlapping strokes overwrite earlier paint, so their order matters.',
    'twins': 'Bring the two movers together. Vertical moves are shared; horizontal moves are mirrored. A wall or boundary can stop one mover while the other continues.',
    'centroid': 'Move the arms until the central body reaches the cross. Left/Right move the first arm; Up/Down move the second left/right when present. There are at most two arms. The body follows their average position; later levels extend their range.',
    'linkage': 'Left/Right retract/extend the horizontal rod; Up/Down retract/extend the vertical rod when present. Match endpoints to target crosses; moving the parent carries its child. There are at most two rods, with a larger range at later levels. In Retract before extending, the parent must be at minimum length before its child can change. The mark beside the child brightens when enabled; a satisfied endpoint goal may need to become false temporarily.',
    'maze': 'Follow connected rooms to the exit cross. Collect visible keys before their matching doors. A bridge must be enabled before entering; disabling it under the mover loses the attempt. Route on a budget has a tighter action allowance.',
    'loops': 'Match the upper token order by rotating either loop forward or backward. The lines below indicate the whole loop and the overlapping short loop.',
    'pegs': 'Tap a peg, then an empty landing two cells away in a straight line. Jump over a peg to remove it; hollow support pieces remain. Finish with exactly one ordinary peg. The white border marks selection.',
    'boxes': 'Move the white player to push blocks onto their matching crosses. Contact and impulse pushes up to two cells while the player stays put. In Deliver only what counts, hollow blocks are obstacles without delivery goals.',
    'agents': 'Bring both movers to their crosses at the same time. Your control moves the first; the other advances on every command, including Wait. Discover the other mover may reverse its direction.',
    'tanks': 'Transfer one unit between adjacent tanks to reach every target-height line. Left/Right transfer on the first connection; Up/Down transfer left/right on the second. There are at most three tanks, with larger capacities at later levels. Liquid is conserved; an empty source or full destination blocks a transfer.',
    'flow': 'Tap the boxed valves on the pipes, then send the signal with Space. Match every outlet’s upper target marks with the lower propagated marks. Branching joins source sets; other profiles swap routes.',
    'gravity': 'Step sideways and let gravity settle the token, or reverse gravity. Reach the cross; walls stop a slide.',
    'merge': 'Tap a block and then another of the same tier to merge them into the next tier. Each tier has an upper tray of hollow required units and a lower tray of filled present units. Match them exactly, including empty trays: both missing and excess units matter. Tap the selected block to deselect; a different tier changes selection.',
    'rules': 'Translate the input glyph row through the displayed dictionaries, top to bottom. Tap each boxed answer to cycle it. In Symbols across appearances, shape identity survives rotation, scale and colour changes.',
    'program': 'Tap the two switch cells in each instruction, read left to right, then run with Space. Match the upper target sequence exactly with the lower emitted sequence. The four examples below show each switch configuration and its concrete output; an empty tray emits nothing. Paired nested brackets show call and return around emissions. Nested recursive calls use the configuration to choose the nesting shown in its example. Examples describe individual instructions, not a solution program.',
    'sigils': 'Tap cells to draw one of the shown sigils exactly. Completing it advances that attribute and clears the drawing. Match each upper target count with its lower current count.',
    'memory': 'Reveal pairs of cards with matching colours. A mismatch briefly shows both cards before hiding them. Matched pairs disappear; remember earlier reveals.',
    'ghost': 'Walk to a plate, then tap the character itself to save the path and return to the start. Space waits one step. Recorded helpers replay as you move or wait, then stay at their endpoints. All plates must be held while you enter the exit corridor.',
    'timing': 'Use Right to advance, Space to wait or Left to retreat toward the exit. Every command advances the clock first; landing on an active hazard loses. Vertical tracks above each hazard align with the clock at the right. A coloured cell marks its dangerous phase, a white inset marks the current phase, and a filled hazard is active. The clock moves downward and wraps along its return edge.',
}


def skill_game(session):
    game = getattr(session.env, '_game', None)
    return game if session.selected['source'] == 'synthetic' and isinstance(game, SkillGame) else None


def human_help(session):
    if session.selected['source'] == 'synthetic' and session.selected['id'] == 'mc01-v1':
        return {
            'title': 'Square Cross Maze',
            'text': 'Move the square onto the cross through the floor corridors. The other background colour marks fixed walls: the whole square must fit through a passage. Six levels introduce horizontal and vertical paths, one turn, two turns, dead ends and loops with alternative routes. Colours vary between seeds.',
            'controls': ['↑ / ACTION1: Up', '↓ / ACTION2: Down', '← / ACTION3: Left', '→ / ACTION4: Right'],
            'later_practice': [],
            'addressing': 'The square is already selected. The cross stays fixed until you reach it.',
            'keyboard': 'Use the arrow keys or D-pad. Each command moves one corridor cell. Click, Space and Undo are unavailable.',
            'cost': 'Every arrow costs one move, even against a wall. The bar shows moves remaining. Reaching the cross on the last move still succeeds. RESET retries the current level after an attempt; after WIN it starts at level 1.',
        }
    game = skill_game(session)
    if game is None:
        return None
    p = game.puzzle
    return {
        'title': p.spec.title,
        'text': CONTEXT_HELP[p.spec.context],
        'controls': control_descriptions(p),
        'later_practice': list(LATER_PRACTICE.get(p.spec.game_id, ())),
        'addressing': 'Use the listed native actions or tap the actual figure, tile or switch. There are no button panels inside the image. Coordinates refer to objects in the native 64 × 64 image; this help stays outside the board.',
        'keyboard': ' '.join(f'{ACTION_LABELS[action]}: {p.commands[command]}.'
                             for action,command in sorted(p.action_commands().items())) +
                    ' Z / Undo restores the previous command state without refunding budget.',
        'cost': 'Every command, miss and Undo costs one action. Blocked commands also cost one. Success on the last permitted action wins. RESET retries this level; a second untouched RESET or a WIN reset starts at level 1.',
    }
