"""Human control documentation. Never imported by an agent or SDK adapter."""
from games.skill_curriculum import SkillGame
from games.skill_core import LATER_PRACTICE

CONTEXT_HELP = {
    'basic_color': 'Tap each filled tile to toggle between two colours. Match its nearby hollow target. Tiles are independent.',
    'basic_unlock': 'Tap an enabled switch to open it permanently. Follow the visible connections: a later switch opens only after every incoming switch is open. Open all switches and their connected goals. Undo can close a switch again.',
    'basic_move': 'Use arrows to move the solid object onto its matching hollow target. Each arrow moves one grid step; the object is already selected.',
    'basic_rotate': 'Tap a piece to turn it clockwise around its fixed white centre. Fit the filled cells into the hollow outline. Each piece turns independently.',
    'basic_reflect': 'Move the solid object with arrows. Its outlined reflection moves across the fixed vertical mirror. Bring the reflection onto the hollow target on the other side. Horizontal reflection reverses left and right.',
    'basic_loop': 'Match the left loop to the hollow reference on the right. The curved arrows turn all tokens one position clockwise or counterclockwise.',
    'basic_transfer': 'Match the liquid in each tank to the small hollow target slots beside it. Direction arrows transfer one unit between adjacent tanks. Empty sources and full destinations block a transfer; liquid is conserved.',
    'basic_pair': 'Tap two cards to reveal their colours. Equal colours stay revealed with a matched underline. A mismatch shows both faces briefly, then hides them. Find every pair. Cards never move during a level.',

    'dials': 'Cycle the lower dials to satisfy the upper targets. In relational games the first dial matches its anchor and = / ≠ links constrain neighbours. Coupled controls also advance the next dial. The small bottom swatches show cycle order.',
    'keys': 'Match each lower attribute to its hollow upper target. In gated games an attribute changes only when its predecessor is at the final cycle value. A swap, when offered, exchanges first and last attributes.',
    'transform': 'Cover every target cross with the shape. Move, rotate or reflect using the labeled controls. Mirror profiles also show a reflected copy; moving-reference profiles let you move the mirror axis.',
    'paint': 'Reproduce the left reference on the right canvas. Each control paints an entire region, overwriting earlier paint. Region diagrams below the boards show the masks in order.',
    'twins': 'Bring the two movers together. Vertical moves are shared; horizontal moves are mirrored. A wall or boundary can stop one mover while the other continues.',
    'centroid': 'Move the arms until the central body reaches the cross. The body follows the average of the arm positions; moving one part changes the whole.',
    'linkage': 'Adjust the rods so their endpoints match the target crosses. Moving a parent carries its children. In Retract before extending, a child can change only while its parent is fully retracted; a satisfied condition may need to become false temporarily.',
    'maze': 'Follow connected rooms to the exit cross. Collect visible keys before their matching doors. A bridge must be enabled before entering; disabling it under the mover loses the attempt. Route on a budget has a tighter action allowance.',
    'loops': 'Match the upper token order by rotating either loop forward or backward. The lines below indicate the whole loop and the overlapping short loop.',
    'pegs': 'Tap a peg, then an empty landing two cells away in a straight line. Jump over a peg to remove it; hollow support pieces remain. Finish with exactly one ordinary peg. The white border marks selection.',
    'boxes': 'Move the white player to push blocks onto their matching crosses. Contact and impulse pushes up to two cells while the player stays put. In Deliver only what counts, hollow blocks are obstacles without delivery goals.',
    'agents': 'Bring both movers to their crosses at the same time. Your control moves the first; the other advances on every command, including Wait. Discover the other mover may reverse its direction.',
    'tanks': 'Transfer one unit between adjacent tanks to reach every target-height line. Liquid is conserved; an empty source or full destination blocks a transfer.',
    'flow': 'Set the valves, then send the signal with Space or the final labeled control. Match every outlet’s upper target marks with the lower propagated marks. Branching joins source sets; other profiles swap routes.',
    'gravity': 'Step sideways and let gravity settle the token, or reverse gravity. Reach the cross; walls stop a slide.',
    'merge': 'Tap a block and then another of the same tier to merge them into the next tier. Meet the exact counts printed above, including zero counts. Tap the selected block to deselect; a different tier changes selection.',
    'rules': 'Translate the input glyph row through the displayed dictionaries, top to bottom. Tap each boxed answer to cycle it. In Symbols across appearances, shape identity survives rotation, scale and colour changes.',
    'program': 'Toggle the two bits of each instruction, then run with Space or the final labeled control. Match the upper output sequence exactly. Values: 0 = empty, 1/2 = emit either symbol, 3 = call F. F emits first symbols then second symbols, as shown in the bottom example. Nested recursive calls instead use each slot’s value as the call depth (0 = no call).',
    'sigils': 'Tap cells to draw one of the shown sigils exactly. Completing it advances that attribute and clears the drawing. Match each upper target count with its lower current count.',
    'memory': 'Reveal pairs of cards with matching colours. A mismatch briefly shows both cards before hiding them. Matched pairs disappear; remember earlier reveals.',
    'ghost': 'Walk to a plate, then use Space / Record ghost to save the path and return to the start. Recorded helpers replay as you move or wait, then stay at their endpoints. All plates must be held while you enter the exit corridor.',
    'timing': 'Advance, wait or retreat to reach the exit. Every command advances the phase first; landing on an active hazard loses. Numbers show hazardous phases and the lower dots show the current phase.',
}


def skill_game(session):
    game = getattr(session.env, '_game', None)
    return game if session.selected['source'] == 'synthetic' and isinstance(game, SkillGame) else None


def human_help(session):
    game = skill_game(session)
    if game is None:
        return None
    p = game.puzzle
    board = getattr(p, 'board_controls', False)
    direct = getattr(p, 'direct_controls', False)
    return {
        'title': p.spec.title,
        'text': CONTEXT_HELP[p.spec.context],
        'controls': list(dict.fromkeys(p.commands)) if direct else [f'{i+1 if board else chr(65+i)}: {command}' for i, command in enumerate(p.commands)],
        'later_practice': list(LATER_PRACTICE.get(p.spec.game_id, ())),
        'addressing': 'Tap pieces or pictorial arrows directly; no numbered or lettered buttons are required.' if direct else 'Tap the board directly; numbers below follow row order.' if board else 'A, B, … label the on-screen buttons, left to right then top to bottom.',
        'keyboard': ('Arrows: ' + ', '.join(p.commands[i] for i in p.arrow_commands) + '. ' if p.arrow_commands else 'Arrows are unavailable. ') +
                    ('Space: ' + p.commands[p.run_command] + '. ' if p.run_command is not None else 'Space is unavailable. ') +
                    'Z / Undo restores the previous command state without refunding budget.',
        'cost': 'Every command, miss and Undo costs one action. Blocked commands also cost one. Success on the last permitted action wins. RESET retries this level; a second untouched RESET or a WIN reset starts at level 1.',
    }
