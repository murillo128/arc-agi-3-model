"""Opt-in scalar issue reports for a manually launched, persistent experiment."""
import json
import logging
from pathlib import Path
import subprocess


def render_report(report, game=None):
    from .online_lora_experiment import GAMES

    def number(value, digits=2):
        return 'N/D' if value is None else f'{value:.{digits}f}'

    games = report['games']
    requested = report.get('requested_games', GAMES)
    diagnostic = report.get('scope') == 'public-training-keyboard-diagnostic'
    pending = [name for name in requested if name not in {g['game'] for g in games}]
    checkpoint = report['checkpoint']
    label = 'Diagnóstico LoRA de teclado (split de entrenamiento)' if diagnostic else 'Ejecución manual LoRA'
    lines = [f"{label}: actualización {'de ' + game['game'] if game else 'final'}.", '',
        f"Código `{report['commit']}`; checkpoint batch **{checkpoint.get('source_batches', checkpoint['expected_source_batches'])}**, "
        f"SHA-256 `{checkpoint['expected_sha256']}`. Origen: {checkpoint.get('release') or 'archivo local indicado en la ejecución'}.", '',
        '| Juego | Niveles | Victoria | Acciones | RHAE % | Delta previo % | Adaptación s | Total s | GPU máx. GiB | Parada |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for row in games:
        lines.append(f"| {row['game']} | {row.get('levels_completed', 0)}/{row.get('total_levels', '?')} | "
            f"{bool(row.get('win'))} | {row.get('actions', 0)} | "
            f"{number(row.get('official', {}).get('rhae_percent'))} | "
            f"{number(row.get('prediction', {}).get('error_percent'))} | "
            f"{number(row.get('diagnostics', {}).get('adaptation_seconds'))} | "
            f"{number(row.get('elapsed_seconds'))} | "
            f"{number(row.get('peak_gpu_allocated_bytes', 0) / 2**30, 3)} | {row['stop_reason']} |")
    lines += ['', f"Estado: `{report['status']}`. Juegos pendientes: {', '.join(pending) or 'ninguno'}.", '',
        f'Agregado de los juegos registrados; RHAE medio sólo al registrar los {len(requested)} juegos solicitados:',
        '```json', json.dumps(report['aggregate'], indent=2, allow_nan=False), '```']
    if game is not None:
        details = {key: game.get(key) for key in (
            'error', 'truncated', 'requests', 'confirmations', 'resets', 'actions_by_level', 'official',
            'prediction', 'prediction_exclusions', 'optimizer_steps_by_module', 'fit_stop_counts',
            'final_fits', 'base_immutable', 'base_frozen', 'inactive_bank_unchanged', 'initial_available_actions')}
        details['numerical_failures'] = game.get('diagnostics', {}).get('numerical_failures')
        details['actions_per_solved_level'] = {key: value['actions']
            for key, value in game.get('actions_by_level', {}).items()
            if int(key) <= game.get('levels_completed', 0)}
        details['first_last_pre_adaptation'] = {key: {k: v for k, v in game.get(key, {}).items() if k != 'actions'}
                                              for key in ('first', 'last')}
        details['window_overlap'] = game.get('window_overlap')
        lines += ['', 'Detalle escalar del juego (errores de ajuste en fracción; las pérdidas posteriores usan datos ya vistos):',
            '```json', json.dumps(details, indent=2, allow_nan=False), '```']
    if 'file_immutable' in checkpoint:
        lines += ['', f"Archivo base intacto al cierre: `{checkpoint['file_immutable']}`."]
    if report.get('error'):
        lines += ['', f"Error: `{report['error']}`."]
    if diagnostic:
        lines += ['', 'Diagnóstico adicional sobre un juego público del split de entrenamiento: '
            'el modelo base puede haberlo visto. No mide generalización a un juego desconocido '
            'ni se incorpora al agregado de los cinco held-out.']
    else:
        lines += ['', 'Comparación descriptiva con #153: cambian tanto los pesos (8246 → 8458) como el protocolo '
            '(pasos/presupuestos e histórico 32 → umbrales/plateau e histórico completo ponderado). '
            'No permite atribuir causalmente diferencias a LoRA. Son juegos públicos held-out.']
    lines += ['', 'N/D incluye denominadores vacíos; '
        'las acciones sin cambios siguen contando, y los resets/cambios de nivel se separan del error predictivo. '
        'Los errores posteriores al ajuste no demuestran generalización. '
        'Pesos de desarrollo; no una puntuación oficial de Kaggle.']
    return '\n'.join(lines) + '\n'


def publish_report(output, report, issue, game=None):
    """Persist the reviewable text first; publication failure never loses results."""
    from .online_lora_experiment import write_json

    name = game['game'] if game is not None else 'final'
    body = Path(output) / f'issue-{name}.md'
    receipt = body.with_suffix('.publication.json')
    if receipt.exists() and json.loads(receipt.read_text()).get('published'):
        return
    try:
        body.write_text(render_report(report, game))
        result = subprocess.run(['gh', 'issue', 'comment', str(issue), '--repo',
            'murillo128/arc-agi-3-model', '--body-file', str(body)], text=True, capture_output=True)
        write_json(receipt, dict(published=result.returncode == 0, returncode=result.returncode,
                                url=result.stdout.strip() if result.returncode == 0 else None))
        if result.returncode:
            logging.error('Issue publication failed (%s); saved report: %s', result.returncode, body)
    except Exception:
        logging.exception('Issue publication failed; local experiment evidence retained')
