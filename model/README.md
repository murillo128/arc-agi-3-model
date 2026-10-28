# Modelo residual latente

## Frozen reconstruction audit

`model.reconstruction_audit` measures encoder/decoder reconstruction separately
from prediction and game completion. It keeps an immutable representation-only
snapshot, CPU exploration weights, evaluation observations and results in a fresh
local directory. It never updates weights or writes observations to training replay.

```sh
python -m model.reconstruction_audit capture --checkpoint RUN/latest.pt --output results/reconstruction
python -m model.reconstruction_audit collect --output results/reconstruction \
  --split configs/splits.public-development.json --environments-dir environment_files
python -m model.reconstruction_audit evaluate --output results/reconstruction --device cuda --pilot
python -m model.reconstruction_audit evaluate --output results/reconstruction --device cuda
python -m model.reconstruction_audit gallery --output results/reconstruction --device cuda
```

All games must be provisioned locally through the official SDK. Collection uses
only public observations/actions and a frozen change head when available (otherwise
visible/uniform exploration). It never selects a private SDK level directly.
Held-out games use whatever levels are reached; training games contribute only
their reserved final level. Terminal frames are excluded, level-entry observations
discard previous-level animation, and ordinary observations retain animation context.
The last levels of synthetic games are not part of this held-out SDK split.

Defaults: two seeds, 4096 actions total per game (including reset actions), two CPU
workers, 300 seconds per game checked between actions, and a uniform reservoir of
up to 64 distinct settled images per game/target. `--actions` can increase the access
budget up to 16384; `--games` selects exact split IDs. Use a separate output directory
for another attempt, copying/linking the same frozen snapshot if comparing access
budgets. Missing games, unreachable levels and timeouts are explicit coverage gaps.
Reports are conditional on the visited states, not coverage of every level or a
representative sample of all possible states.

CUDA evaluation keeps original weight precision, uses BF16 autocast when supported,
one spatial frame chunk and no predictor/optimizer. `--cuda-memory-fraction` defaults
to 0.08 of device memory; this limits the PyTorch allocator, not driver/context or
other processes. Check the one-image pilot and actual device free memory before
running alongside training. `--device cpu` is available if coexistence is unsuitable.
The report gives inference wall time, peak allocated/reserved memory and weight
integrity; model loading and the final integrity hash are outside inference timing.

Pixel error uses all evaluated pixels. Balanced color error averages per-target-color
error ratios after pooling frames; non-dominant error excludes all tied largest colors
in each image, without assigning a particular palette entry to background. Changed
error uses real before/after changed pixels in same-level observations, excluding
resets and level entries. A zero denominator is unavailable, not perfect. Duplicate
settled images count once even if their animation context differs. Results also
include perfect-frame fraction, per-color counts, access coverage and worst images.
Keep captured observations, snapshots, galleries and detailed local results out of Git.

## Optional spatial predictor growth and Undo exclusion

Sequence exploration accepts `"allow_undo": false` inside `sequence_exploration`.
It removes ACTION7 before uniform, mixed or learned collection samples an action.
The SDK observation, manual controls and game rules retain their original actions.
An observation with no remaining ordinary action raises an explicit error; the
collector never silently re-enables Undo. The default remains `true` for old runs.

`spatial_adapter_config.example.json` enables `network.rotation_adapter_width: 16`.
The default network value is zero (disabled); a positive width requires local
dynamics and pixel latents. A small predictor branch shares 5×5 filters across four
quarter-turn orientations, retains all four feature sets, and adds native click
and relative-coordinate maps plus action/history conditioning. A dilated depthwise
convolution and learned orientation mixing produce an additive latent correction.
This does not assume that arbitrary game rules or pretrained latents are equivariant.

Only the final correction projection starts at zero. Thus
`new_delta = existing_delta + residual_scale * correction` initially returns the
existing prediction exactly. The inner filters initialize normally: the output
projection can learn on the first update and propagate gradients inside thereafter.
A matrix of ones would not preserve the prediction, and zeroing the entire branch
would prevent useful feature learning. This preserves the initial function, not a
guarantee that subsequent optimization improves every game.

For a sequence checkpoint, copy its exact initialization configuration and set
only `rotation_adapter_width`; keep all other network/training settings unchanged.
Set `allow_undo: false` in its training configuration and resume once with:

```sh
python -m model.sequence_training --training-config CONFIG --output RUN \
  --resume --disable-undo --upgrade-predictor
```

Either migration flag may also be used separately. Existing weights, named AdamW
moments and effective learning rates, representation/exploration optimizers, phase,
counters and RNG survive. Only new adapter parameters start without AdamW history;
Undo examples are filtered from auxiliary exploration replay. Unrelated configuration
changes fail closed. A trained predictor must retain every old AdamW entry and its
required fields; empty optimizer state is allowed only before its first update.
Click maps recover integer SDK coordinates before checking native frame bounds.
Once saved, ordinary `--resume` restores the expanded model.

The adapter's native Gaussian click map has unit peak and a default fixed width
of 1.5 pixels. Set `network.learnable_click_sigma: true` in the initialization
configuration to learn one shared width from the predictor loss. It requires an
enabled rotation adapter. The parameterization
`sigma = 1.5 * exp(log(4) * tanh(log_scale))` starts with `log_scale=0`, preserving
the exact old map, and smoothly bounds sigma to 0.375–6 pixels. Click coordinates,
relative-coordinate channels, padding masks and peak intensity stay unchanged.
The scalar joins the predictor's existing no-decay AdamW group at its current LR.

For an existing fixed-width checkpoint, copy its configuration, enable only that
switch, and resume once with `--upgrade-predictor`. Existing weights and AdamW
states are retained, including an already-trained rotation adapter; only the new
scalar starts without optimizer history. Subsequent resumes use ordinary
`--resume`. The dashboard reports sigma in native pixels and fixed/learned status;
each batch records `click_sigma_pixels` after its weight updates. This adds one
parameter and does not itself establish better prediction accuracy.

To return a learned-width run to **fixed sigma=1.5**, copy its initialization
configuration, change only `network.learnable_click_sigma` to `false`, and resume
once with `--fix-click-sigma`. This removes only the width scalar and its AdamW
state. All other weights (including trained spatial filters), optimizer moments,
learning rates, counters, phase and RNG are retained. It cannot be combined with
other migrations. After saving, use ordinary `--resume`. The fixed-width map no
longer backpropagates through a width parameter; predictions may change because
the learned width is deliberately replaced with 1.5, rather than frozen at its
last value. Dashboard metadata reports the fixed width.

The old report is archived at the adaptation boundary; new charts start with the
new action population. The dashboard reports adapter width and actual Undo counts.
`architecture.json` describes the width-16 example and includes its tensor bindings.

## Secuencias con memoria entre niveles

El modo optativo `model.sequence_training` entrena el predictor con **sesiones
independientes** (cuatro por batch en el ejemplo base). Una sesión conserva el contexto recurrente
durante sus niveles y reinicios, hasta su presupuesto (512 acciones en el ejemplo).
Al comenzar otra sesión, incluso del mismo juego, la memoria empieza en cero.
Los pesos se comparten entre todos los juegos; el predictor no recibe su nombre
ni reglas internas. Las predicciones imaginadas nunca actualizan esta memoria real.

Los juegos sintéticos permutan los 16 colores, incluidos fondo, HUD, pistas y
objetivos, sin cambiar sus reglas ni fusionar colores. Durante una sesión de
secuencia se conserva la misma permutación en todos los niveles y reinicios,
aunque cambien las escenas; la siguiente sesión elige otra a partir de su semilla.
Los juegos oficiales del SDK conservan sus colores originales.

```sh
.venv/bin/python -u -m model.sequence_training \
  --training-config model/sequence_config.example.json \
  --source model/runs/curriculum/current/latest.pt \
  --output model/runs/secuencias --device cuda
.venv/bin/python -u -m model.dashboard \
  --sequence-report model/runs/secuencias/report.json --port 8767
```

`--source` importa pesos sin modificar el archivo original ni sus optimizadores.
Sin él, `init_config` inicializa pesos nuevos. `--max-batches 2` acota una prueba;
`--resume` sustituye a `--source` para restaurar pesos y momentos del optimizador
desde el directorio de salida. Por defecto se exige el mismo plan al reanudar. SIGINT/SIGTERM
terminan el batch actual y guardan atómicamente; no se serializan entornos vivos.
Una interrupción brusca puede repetir el batch aún no guardado. Se conserva un
checkpoint por batch completo; el presupuesto de acciones puede sobrepasarse
hasta completar ese último batch. Este modo no declara convergencia ni publica pesos.

`--unlimited` elimina el límite total de acciones: sigue entrenando hasta una
interrupción manual o hasta `--max-batches`, si se especifica. Se puede activar
con `--resume` incluso cuando el presupuesto anterior ya se ha agotado, conservando
pesos, optimizadores, contadores y plan. Es una opción de cada lanzamiento: hay
que incluirla al reanudar; sin ella vuelve a aplicarse `settings.actions`.
El dashboard muestra **Sin límite de acciones** (`budget: null` en el informe).
Los límites por sesión, nivel, historial y diagnóstico permanecen activos.

```sh
.venv/bin/python -u -m model.sequence_training \
  --training-config model/sequence_config.example.json \
  --output model/runs/secuencias --device cuda --resume --unlimited
```

To retune only the predictor in an existing run, add
`--predictor-learning-rate 1e-4` to the resume command. The override is applied
after restoring Adam's state: it changes no weights, moments, counters,
encoder/decoder rates, phase gates or telemetry identity. The saved optimizer
retains the effective rate on subsequent resumes without the flag; the
checkpoint's training metadata and dashboard report show that effective rate.
The original plan remains the resume compatibility contract. Its settings are
not silently edited, and the override does not permit other plan changes.

For a paired experiment, stop and back up the live run first, then use:

```sh
.venv/bin/python -m model.compare_predictor_lr \
  --training-config model/sequence_mixed.example.json \
  --checkpoint /path/to/immutable-backup.pt \
  --output model/runs/predictor-lr-comparison --device cuda
```

Both trials restore the same complete checkpoint and optimizer state, consume
one shared real corpus covering every configured training game, and preserve
ordinary representation updates and their learning rates. The default probe is
a separate 128-step session per game, with disjoint seeds and no optimization;
these probes do not control the live curriculum. The report distinguishes
pixel-pooled and equal-game delta error, synthetic/SDK groups, and full-frame NLL.
Its paired game bootstrap describes variation over this corpus, not repeat-run
uncertainty. A candidate must improve both overall error summaries without
worsening either source group's pooled error. This command never applies a
candidate to the live run; inspect its evidence before using the explicit
override. Keep the generated corpus and checkpoints outside Git.

The [recorded 91-game comparison](predictor_lr_comparison_evidence.json) retained
`3e-5`: after 384 matched predictor updates, separate-probe pooled delta error was
92.21%, versus 95.60% at `1e-4`. This is evidence for that checkpoint and corpus,
not a universal optimal rate; the live run resumed its original weights and rates.

La configuración reutiliza los juegos, niveles, rutas y splits del plan anterior,
añadiendo `sequence`. `objective` debe ser `world_model` (es su valor por defecto
en este modo). `sequence_control` activa fases de representación y predictor con
umbrales sobre los batches normales de entrenamiento. Guarda cada batch completo;
no activa el planner ni modifica la rotación configurada de niveles. Los planes
anteriores sin `sequence_control` mantienen su comportamiento y evaluación fija.

Once a controlled run has entered the predictor phase, resume with
`--freeze-representation` to stop encoder and decoder optimization completely.
The operation preserves model weights, both optimizer states, counters, RNG and
the existing plan; it cannot be combined with another adoption, curriculum change
or predictor LR override. Save a recovery checkpoint before changing a live run.
The checkpoint records the completed batch at which representation was frozen;
ordinary `--resume` retains that freeze without repeating the flag. Older
checkpoints remain unfrozen. A frozen run rejects subsequent curriculum/adoption
changes instead of implicitly unfreezing representation, with the explicit
teacher-guided planning extension described below as the only exception.

Predictor training retains its visual loss and differentiates through the frozen
decoder. Reconstruction and prediction diagnostics continue on normal training
batches. Representation loss is absent (not zero), and redundant encoder-drift
measurement is skipped with an explicit frozen status in the dashboard. There
is no automatic unfreeze; use the preserved checkpoint to resume the earlier
training regime. Freezing is not evidence that prediction quality improved.

An optional `sequence_planning` section adds learned visual goals and a
success-first path head alongside the predictor, initially supervised only by
the bundled Follow the Cross teachers. Adopt it once with `--adopt-planning` on
a frozen predictor run; ordinary resume preserves both new heads and all existing
weights and optimizer states. See [sequence planning](../docs/sequence-planning.md)
for the configuration, real-outcome labels, monitoring and deployment boundary.

### Mixed synthetic and public SDK training

[`sequence_mixed.example.json`](sequence_mixed.example.json) combines all 71
synthetics with the 20 public SDK development games. The pinned public split in
[`splits.public-development.json`](../configs/splits.public-development.json)
uses seed 42: `ar25`, `cn04`, `lf52`, `tn36`, and `vc33` stay held out completely.
The level caps come from the SDK's public `win_levels`, excluding each training
game's final level. Provision these public games in `environment_files` with the
pinned SDK before running the offline configuration. No game implementations
or downloaded frames are distributed with the configuration.

Sessions start at the SDK's native first level and reach later allowed levels
through real actions. A configured upper level is permission to collect it,
not evidence that it has been reached. Entering a reserved level stops collection
before its target enters training. Batches still contain up to eight independent
512-action sessions; 24 batches cover two shuffled rounds of the expanded catalog.

To append games to an existing run without resetting learned weights, optimizer
moments, RNG or counters:

```sh
.venv/bin/python -u -m model.sequence_training \
  --training-config model/sequence_mixed.example.json \
  --output model/runs/secuencias --device cuda --resume --extend-curriculum --unlimited
```

First stop the existing trainer gracefully so it finishes and saves its batch.
`--extend-curriculum` only permits appending stages after the unchanged existing
stages, adding the first SDK configuration, and increasing the evidence window
to cover the larger catalog. Other settings, thresholds, existing SDK splits,
stage edits/removals/reordering, and phase adoption remain incompatible.
New stages begin without qualification evidence. Existing evidence is retained;
an already admitted predictor returns to representation qualification for the
expanded curriculum. The extension boundary is recorded in the checkpoint and
report, while historical batches remain identifiable as the earlier catalog.
Once the extended checkpoint is saved, ordinary `--resume` works again; leaving
`--extend-curriculum` on with the identical plan is also safe.

To remove games or levels, stop the trainer gracefully, remove only those entries
from its configuration, then resume with `--resume --restrict-curriculum`.
The retained stages must keep their order and exact definitions; learning rates,
thresholds, SDK splits and other settings cannot change in the same operation.
Weights, optimizer moments, counters, RNG and the current training phase survive.
Phase evidence is filtered to retained stages. The exploration head also survives;
its unlabelled replay buffer is cleared so removed games are no longer rehearsed.
The previous report is archived as `report-before-curriculum-restriction-BATCH.json`,
and dashboard history and optimization measurements start a new segment for the
smaller catalog. The checkpoint and report record the restriction boundary.
After the first save, ordinary `--resume` works; leaving the flag on with the same
plan is idempotent.

### Fases sobre los datos de entrenamiento

El ejemplo incluye `sequence_control: {"window_batches": 4, "diagnostic_samples": 128,
"decoder_learning_rate_after_predictor": 3e-7}`. Cada batch se mide **antes** de
actualizar sus pesos, con memoria causal y las mismas transiciones que después
se entrenarán. No se recogen partidas adicionales ni se reutilizan trayectorias
fijas. Estas métricas controlan la optimización; no son una evaluación de
generalización. La muestra de diagnóstico se reparte entre juegos/niveles y
reserva hasta la mitad de cada cupo para cambios reales observados, incluidos
clics poco frecuentes. El límite debe cubrir todos los pares presentes.

Primero solo aprende representación. Para cada juego/nivel se acumulan los
numeradores y denominadores de los últimos `window_batches`, sin promediar
porcentajes de escenas de tamaños distintos. El error de admisión reconstruye
la observación real posterior y cuenta errores sobre la unión de sus cambios
reales y reconstruidos respecto a la observación anterior. El fondo intacto no
diluye el error. La reconstrucción global de todos los píxeles sigue disponible
como diagnóstico separado. La deriva es `sqrt(sum((z_después-z_antes)^2) /
sum(z_antes^2))` sobre exactamente las mismas imágenes del batch, alrededor de
las actualizaciones de representación; no compara escenas distintas.

Se reutilizan los umbrales por etapa `representation_open` (0,10), `latent_drift`
(0,15), `minimum_samples` (8), `minimum_changed` (4) y `patience` (3). Cada par
necesita nuevas muestras para sumar un batch apto. Primero deben haberse observado
todos los niveles sintéticos y el primer nivel de cada juego SDK. Los niveles SDK
superiores se incorporan desde su primera muestra supervisada real; los aún no
alcanzados se marcan como tales y no se consideran aprobados. La cobertura
observada persiste aunque las muestras salgan de la ventana o se reanude el proceso.
Los pares ya observados con menos muestras, cambios o muestras de deriva que los
mínimos se **ignoran para la admisión**, conservando sus métricas y sin sumar
aprobados. Siguen recogiéndose y entrenándose normalmente. En cuanto reúnen datos
suficientes vuelven a participar en la condición de paso. Debe haber al menos un
par elegible y todos los elegibles deben cumplir los umbrales y `patience`;
un error medido, una escala latente inválida o confirmaciones pendientes siguen
bloqueando. Una ventana completamente insuficiente no activa el predictor.
Los checkpoints anteriores recuperan la cobertura conocida desde su ventana,
contadores positivos y niveles SDK observados, sin alterar pesos ni optimizadores.
La admisión activa el predictor
en el **siguiente** batch y se conserva; una regresión posterior permanece visible.
`prediction_open` y `success` no activan un planner en este modo.

En representación se usa `settings.representation_learning_rate` en ambos
módulos. En predictor se usa `settings.encoder_learning_rate_after_predictor`
para el encoder y `sequence_control.decoder_learning_rate_after_predictor` para
el decoder (3e-7 en ambos en el ejemplo), manteniendo predictor a 3e-5. Se
conservan pesos y momentos. Fase, ventana y contadores se guardan al completar
el batch y se recuperan con `--resume`.

Para adoptar las fases en un checkpoint antiguo, guardar primero una copia
local del checkpoint/configuración y añadir `sequence_control` al plan. La única
otra modificación permitida es el LR posterior del encoder. Ejecutar una vez
`--resume --adopt-training-phases --max-batches 1` (añadir `--unlimited` si se
había agotado el presupuesto), y después reanudar normalmente **sin** el flag
de adopción. No reinicia pesos, momentos ni contadores; comienza en representación
con evidencia nueva y archiva el informe previo como
`report-before-training-phases-N.json`. Cambios en juegos, pérdidas o arquitectura
siguen rechazados. El dashboard mantiene separados los dos tipos de medición.

| Ajuste `sequence` | Función |
| --- | --- |
| `batch_games: 4` | Sesiones independientes por batch; pueden repetir tipo de juego con semillas diferentes. |
| `session_steps: 512` | Presupuesto de movimientos de cada sesión. |
| `block_steps: 16` | Pasos que aportan pérdida en cada actualización, además del contexto previo con gradiente. |
| `context_steps: 16` | Contexto previo adicional con gradiente y sin repetir su pérdida; permite aprender a través de cambios de nivel alineados con el corte del bloque. |
| `level_steps: 32` | En sintéticos, acciones antes de pasar al siguiente nivel configurado; después se vuelve al primero. Si solo hay un nivel, se reinicia al agotar cada cuota. Un fracaso reinicia el mismo nivel hasta consumir su cuota. |
| `game_schedule: "interleaved"` | Reparto circular entre juegos; `blocked` dedica cada batch a un tipo de juego. `shuffled` baraja cada vuelta y visita cada juego una vez, con un último batch menor si es necesario. |
| `loss_start_weight: 1.0` | Peso de la primera acción: 1 mantiene la pérdida uniforme; 0.25 o 0.5 prueban una rampa. |
| `loss_warmup_steps: 32` | Acciones hasta alcanzar peso 1, contado desde el inicio de la sesión, sin reiniciar en cada nivel. |
| `representation_updates: 4` | Actualizaciones separadas de reconstrucción por batch; 0 congela encoder/decoder para comparar el predictor. Se rotan sesiones y se elige un par real por actualización. |

El ejemplo [`sequence_all_synthetic.example.json`](sequence_all_synthetic.example.json)
selecciona los **71 juegos sintéticos y sus 424 niveles**, con `objective: world_model`.
Usa hasta ocho sesiones de 512 acciones: cada vuelta tiene ocho batches de ocho
juegos y uno de siete. El orden se baraja por semilla de entrenamiento y vuelta;
el contador de batches del checkpoint determina exactamente la continuación.
Las acciones mantienen su orden causal y los niveles conservan el orden configurado.
Los modos anteriores y sus checkpoints siguen usando su reparto original.

Optional `sequence_game_weights` changes the number of independent sessions per
game in a shuffled round. For example, alongside `sequence`:

```json
"sequence_game_weights": {
  "bf02-v1": 3,
  "bf04-v1": 3,
  "bf08-v1": 3,
  "mc01-v1": 3
}
```

Omitted games keep one session. Values must be integers in 1..16 and refer to
configured games; only `game_schedule: "shuffled"` supports this option. A catalog
of 31 games with these four weights has 39 sessions per round, split into batches
of 8/8/8/8/7. Repeated games can share a batch but have different seeds and separate
memory. The per-session loss formula does not change. These are session quotas,
not guaranteed transition shares: an SDK session can end early at its access boundary.

To change an existing run, back up its checkpoint and configuration, then resume
once with `--adopt-game-weights`. This allows only the frequency change, retaining
weights, optimizers, exploration replay, phase and counters. It cannot combine with
other migrations or an LR override. A new shuffled round starts at the saved batch
boundary; later ordinary `--resume` uses the persisted boundary. Default unweighted
plans retain their previous order. The dashboard's Curriculum → Game exposure table
shows planned frequencies and actual session/transition counts since adoption;
earlier chart history is retained. Compare per-game errors across the boundary,
since changing the game mix also changes aggregate metrics.

Las ocho actualizaciones de representación incluyen cada sesión del batch, también
en el batch corto (una sesión recibe una actualización adicional). Las 384 muestras
de diagnóstico permiten al menos ocho por juego/nivel presente si hay suficientes
transiciones reales; los cambios observados siguen siendo obligatorios. La ventana
de 18 batches equivale a dos vueltas y evita perder evidencia entre visitas. Se mantienen
los límites de calidad, el mínimo de evidencia y tres visitas aptas nuevas por cada
par; una media global favorable no abre el predictor. Con más juegos, esta primera
fase puede durar más y no se garantiza convergencia.

Para iniciar este experimento desde pesos y optimizadores nuevos, usa una salida nueva:

```sh
.venv/bin/python -u -m model.sequence_training \
  --training-config model/sequence_all_synthetic.example.json \
  --output model/runs/sequences-all-synthetic --device cuda --unlimited
```

Conserva el directorio y la configuración del entrenamiento anterior. `--resume`
se añade únicamente al retomar un checkpoint del nuevo experimento; no uses
`--source` para esta inicialización desde cero. El número de sesiones es un límite
por batch, no una acumulación de gradientes entre los 71 juegos.

En cada bloque, el predictor recibe los latentes reales anteriores y sus acciones;
el resultado real siguiente solo se incorpora después de predecir. Los saltos de
nivel son contexto con un marcador de reinicio, no objetivos del delta. Cada
entrada del batch tiene memoria propia, agrupada por geometría nativa sin mezclar
estados. Antes de un bloque se reconstruye su prefijo con los parámetros actuales
y sin gradiente hasta el comienzo del contexto adicional. El contexto y el bloque
sí propagan gradiente por la recurrencia; solo el bloque aporta nuevos términos de pérdida.
Esto evita llevar estados obsoletos entre actualizaciones, pero tiene coste creciente
con la sesión y **no equivale a retropropagar por las 512 acciones completas**.

La pérdida conserva MSE latente y NLL visual. Se ponderan los pasos válidos y se
normaliza sobre la **sesión completa**, dando el mismo peso a cada sesión; padding
y saltos no cuentan. La media de los objetivos de los bloques equivale a esa
media por sesión. Normalizar cada bloque por sus propios pesos cancelaría buena
parte de la rampa. Encoder/decoder siguen aprendiendo reconstrucción por separado,
con las tasas de su fase (en predictor, encoder y decoder 3e-7 en el ejemplo); no reciben el
gradiente predictivo. Se codifica de nuevo después de esas actualizaciones y los
latentes se conservan únicamente mientras el encoder permanezca fijo.

En el SDK los niveles permitidos deben ser el prefijo `[1, 2, ...]`: se recorren
mediante acciones públicas, sin forzar avances ni usar profesores. Una transición
que entra en un nivel reservado/no permitido se excluye entera y termina la sesión.
Sin control de fases, la evaluación usa semillas disjuntas y contexto propio.
Con control de fases se miden los batches de entrenamiento antes de actualizarlos.
El dashboard separa error inicial,
tardío y por juego, TPS del último batch completo y coste de juego/codificación/
predictor/evaluación/guardado. Error de predicción no es éxito en el juego.

Los agentes que utilicen estos pesos deben activar
`AgentConfig(memory_across_levels=True)`. Esa opción se conserva en el checkpoint
portable del agente; el valor por defecto sigue siendo `False` para los agentes
anteriores. `Agent.reset()` inicia una nueva sesión y siempre vacía la memoria.

Ensayos breves en RTX 4070 Ti (BF16, cuatro hilos CPU), desde los pesos existentes:
con bloques de 8/16/32/64 y 16 pasos de contexto, el predictor procesó
69/79/84/88 transiciones/s, con 4,15/4,23/4,37/4,64 GiB asignados. Se eligió
16 como punto inicial: error medio por juego del 70,08 %, frente al 70,80 %
con 32 y 72,56 % con 64. Los bloques mayores también hacen menos actualizaciones
para las mismas transiciones; esto no establece un óptimo independiente del LR.
Las rampas desde 0,25 y 0,5 apenas cambiaron el error (hasta 0,1 puntos), por lo
que el ejemplo conserva pesos uniformes.

Con encoder/decoder activos y las mismas 4.096 transiciones, alternar juegos
redujo el error medio del 83,50 % al 77,74 %; agrupar cada juego por separado lo
dejó en 86,44 %. El recorrido completo midió 50 acciones/s incluyendo evaluación
y guardado, con pico de 5,36 GiB. Son ensayos cortos: **borrar el historial aún
mejoró la evaluación al 74,46 %**, por lo que no demuestran que el modelo ya
aproveche la memoria para inferir reglas. Los resultados y límites están en
[`sequence_training_evidence.json`](sequence_training_evidence.json).

## Learned interaction exploration

The mixed SDK/synthetic sequence example enables `sequence_exploration.policy:
"learned"`. Omitting this section preserves the legacy uniform collector. The
auxiliary CPU head estimates whether a real action changes the next image; it
uses palette-invariant image statistics, local pixel neighborhoods, the legal
action mask and previous real-action/change context. It does not backpropagate
into the world model or replace the planner's success/duration heads.

Action IDs remain uniform among the currently allowed IDs. For clicks, 25% stay
uniform over native SDK coordinates 0..63. The rest sample visible-color
candidates using learned change scores. Repeated confirmed no-op coordinates
receive a temporary penalty on the same image, but are never banned. Flat or
missing frames use uniform clicks. The predictor keeps every real transition,
including no-ops; only reset/terminal/level-boundary responses are excluded from
the auxiliary head's binary targets. Timers can still produce positive labels:
this estimates **visible change**, not success or information gain.

The head learns from a bounded recent feature replay after each batch and saves
its own weights, Adam state and sampling RNG in `latest.pt`. To add it to an
existing uniform run, add the section to a copy of the same training config and
resume once with `--adopt-exploration`. An optional `--exploration-source HEAD.pt`
loads only a separately bootstrapped auxiliary head. Later restarts use ordinary
`--resume`; repeating adoption or changing other plan settings is rejected.
World weights, existing optimizers, LRs, counters, phase, report history and the
optimization projection identity are retained. Uniform fixed evaluation, when
configured, remains uniform. The dashboard marks the collection change and
shows changed actions/clicks, auxiliary loss, time and per-game counts; comparing
predictor errors across this boundary also changes the sampled distribution.

`python -m model.compare_sequence_exploration --help` provides a CPU comparison
of uniform, visible-element and learned click collection. It accepts a trusted
local real training corpus (only its `training` sessions) or an auxiliary head;
`--bootstrap-mixed` adds separate real training seeds before comparison. Large
corpora and head checkpoints belong in ignored run directories, not Git.
`PathPlanner(..., change_prior=explorer)` optionally reuses the head to propose
click candidates, keeping uniform candidates and every legal non-click action.
Goal-success-first route ranking is unchanged; this prior does not start or
train the goal planner in sequence mode.

## Monitorización completa e histórico del planner

El dashboard de secuencias conserva las secciones de representación/predictor,
planner, evaluación congelada y currículo, junto a las vistas de optimización.
Muestra las pérdidas del encoder/decoder y sus componentes, MSE latente y pérdida
visual del predictor, desglose del delta, diversidad/rango/coordenadas casi
constantes, sensibilidad a cambios, deriva, ejemplos visuales y tiempos por batch.

El desglose del delta reutiliza las predicciones de la evaluación fija de cada
batch. Los diagnósticos adicionales se calculan en el primero tras arrancar o
reanudar y luego cada cuatro batches: hasta 32 transiciones y 64 estados reales,
repartidos determinísticamente en las sesiones fijas. Reconstrucción y rollout a
tres pasos usan esta muestra; el error a un paso usa todas las transiciones.
Los rollouts reciben la memoria causal real de su origen y encadenan latentes
predichos con las acciones registradas; no atraviesan reinicios ni cambios de
nivel. No representan rutas seleccionadas por el planner. La deriva relativa
usa la primera medición del proceso como referencia, que se indica en la gráfica
y se renueva al reanudar. Se conserva un máximo de 200 batches y únicamente la
última galería de hasta ocho ejemplos, fuera de los checkpoints. Las métricas
nuevas no reconstruyen retrospectivamente datos ausentes.

El dashboard puede conservar un historial numérico más largo **sin reiniciar el
trainer**. Esta persistencia es optativa y pertenece al dashboard:

```sh
.venv/bin/python -m model.dashboard \
  --sequence-report model/runs/secuencias/report.json \
  --sequence-history model/runs/secuencias/dashboard-history.json \
  --max-points 2000 --port 8767
```

El archivo guarda como máximo 2.000 batches en este ejemplo, sin imágenes ni
pesos; se actualiza atómicamente cuando cambia el informe, aunque no haya un
navegador abierto. El proceso comprueba el informe cada cinco segundos. Se inicia
con los batches todavía disponibles: no recupera los ya descartados. Cambiar de
ejecución/cohorte o retroceder el contador evita reutilizar registros incompatibles.
Sin `--sequence-history` el dashboard sigue sin escribir datos. Si no se puede
guardar el archivo, avisa y mantiene las mediciones actuales. No se debe reutilizar
el archivo entre varios servidores simultáneos.

**Chart history** permite ver 200, 1.000 o todos los batches retenidos. La tarjeta
**Prediction trend** calcula ventanas de 50 o 100 batches: primero agrega los
conteos `wrong / union` de cada juego y luego promedia los errores con igual peso.
La línea **Pooled pixels** pondera por el total de píxeles de esa unión. Una
cohorte incompleta o sin denominador no se muestra como error cero. La comparación
usa las dos últimas ventanas completas consecutivas, indica el cambio en puntos
porcentuales y permite desplegar el detalle por juego. Los huecos y cambios de
fase, LR, fuente de medición o política de exploración interrumpen las ventanas.
El zoom vertical es optativo y mantiene los valores del eje visibles. Estas
mediciones no son tasas de victoria ni pruebas de generalización.

El historial viaja por `api/sequence-history`, con ETag, únicamente cuando cambia;
`api/state` conserva su informe reciente. El dibujo limita los marcadores por
serie, pero las medias usan todos los datos de su ventana. Los contadores y
ejemplos visuales siguen mostrando el último batch, independientemente del rango.
La identidad `history_run` del informe servido se calcula a partir de su ruta,
proceso (`optimization_identity`) y cohorte. El navegador sólo incorpora el
archivo cuando coincide con esa identidad; si el colector todavía está leyendo
otra ejecución, muestra únicamente los batches actuales hasta que se actualice.
Las gráficas de diagnósticos periódicos conservan su EMA entre muestras cuando
los batches intermedios existen; los huecos reales y cambios de régimen sí la
interrumpen.

El modo de secuencias recoge acciones aleatorias: no inventa métricas de éxito o
pérdidas de un planner inactivo. Para consultar todas las gráficas y trayectorias
de una ejecución anterior, sin mezclarlas con las actuales:

```sh
.venv/bin/python -m model.dashboard \
  --sequence-report model/runs/secuencias/report.json \
  --history-subgoal-report model/runs/curriculum/current/report.json --port 8767
```

**Histórico del planner** abre las vistas completas anteriores con su diseño
compartido y una marca de solo lectura. Sus endpoints `api/history/state` y
`api/history/trajectory` son independientes de `api/state`; sus controles de
currículo están desactivados. Las vistas antiguas con `--subgoal-report` siguen
funcionando de manera independiente. No se necesitan pruebas visuales para las
comprobaciones de contratos HTTP, datos y runtime de JavaScript.

## Observar gradientes y desplazamientos en directo

El modo de secuencias admite `--optimization --optimization-every 8`, también
con `--resume`. La instrumentación se activa sin cambiar el plan ni los
checkpoints. El dashboard muestra la pestaña **Optimización** desde cualquier
página; las ejecuciones no instrumentadas muestran un estado explícito de ausencia.
No se reconstruyen gradientes históricos a partir de los pesos.

```sh
.venv/bin/python -u -m model.sequence_training \
  --training-config model/sequence_config.example.json \
  --output model/runs/secuencias --device cuda --resume \
  --optimization --optimization-every 8
.venv/bin/python -m model.dashboard \
  --sequence-report model/runs/secuencias/report.json --port 8767
```

- **Pulso:** RMS del gradiente antes de recortarlo, factor global de recorte,
  LR del grupo y desplazamiento real tras `optimizer.step()`. La magnitud y
  alineación del desplazamiento usan hasta 256 coordenadas fijas por tensor;
  `RMS(ΔW)/max(RMS(W), 1e-12)` es una estimación sobre esa muestra. Los mapas
  de la matriz seleccionada calculan RMS sobre bloques completos, sin cancelar
  signos. El histograma de ΔW utiliza la muestra. Se distinguen congelación,
  gradiente ausente y cero medido. El último update de cada batch siempre se
  captura, además de la cadencia, para alinearlo con su evaluación.
- **Trayectoria:** PCA por módulo sobre la misma muestra estratificada, con dos
  ejes fijos calibrados tras cuatro batches. La altura es la NLL de color sin
  ponderación sobre las observaciones reales de evaluación, idénticas entre
  batches; no es el loss del minibatch ni un paisaje de pérdidas. El porcentaje
  visible y el giro pertenecen a la muestra, no al vector completo. Recalibrar
  cambia la base y reproyecta todos los puntos retenidos; no mueve los pesos.
- **Memoria:** sonda inicial y después bajo demanda. Derivada de la pérdida
  visual del último paso supervisado de una sesión respecto a sus estados de
  memoria, con el mismo corte de contexto/bloque que el entrenamiento. Muestra
  cambios de nivel, pérdida con memoria y pérdida con la memoria entrante borrada.
  Es un diagnóstico local, no una medida global de utilidad. `autograd.grad`
  no escribe gradientes de parámetros y se restauran modos, permisos y RNG.

Los controles de captura se atienden al iniciar el próximo batch. Al pedir otra
matriz, el siguiente update que la entrene registra el detalle; no se inventa el
detalle de instantes anteriores. La selección fijada conserva su snapshot aunque
lleguen nuevos batches. Los datos se publican al guardar el batch completo.
Si un punto de la trayectoria ya no conserva su captura de optimización, el
inspector indica «Captura no disponible» y vacía sus medidas y mapas. El transporte
valida también los registros internos y la correspondencia entre puntos y
proyecciones; un registro malformado vacía la vista y se vuelve a consultar hasta
recibir datos válidos, sin conservar una revisión inválida en la caché.

En el directorio del run se conservan `optimization.json` (64 capturas y hasta
96 puntos), `optimization_samples.npz` (muestras/bases, sin pickle) y, mientras
esté pendiente, `optimization_request.json`. Son artefactos locales ignorados,
no pesos adicionales. La reanudación exige una identidad compatible y descarta
telemetría posterior al checkpoint restaurado. HTTP limita el tamaño, usa un
token por servidor para las solicitudes y respuestas condicionales para no
reenviar varios MB cada dos segundos. El servidor sigue sin importar Torch.

El diseño común de navegación, tarjetas, tablas y controles se aplica a las páginas
de secuencias, seguimiento general, campañas, subobjetivos y optimización.
Validación funcional y numérica sin navegador; la apariencia la verifica el usuario.

## Entrenamiento configurable

`model.subgoal_training --training-config ARCHIVO.json` permite elegir juegos,
niveles, orden y umbrales sin editar código. Empieza con
[`training_config.example.json`](training_config.example.json); el ejemplo
[`training_config.sdk.example.json`](training_config.sdk.example.json) combina
un sintético con el SDK. Las rutas internas son relativas al archivo JSON.
La configuración de red sigue en `init_config`; todos los componentes se
inicializan desde cero. Sin `--training-config`, la CLI anterior sigue disponible.

```sh
.venv/bin/python -m model.subgoal_training \
  --training-config model/training_config.example.json --validate-config
.venv/bin/python -u -m model.subgoal_training \
  --training-config model/training_config.example.json \
  --output model/runs/mi-entrenamiento --device cuda
```

La primera orden solo valida y muestra el plan; no construye el modelo ni abre
un entorno. La segunda exige un directorio vacío y toma un bloqueo exclusivo.
`--device cpu` permite usar una red pequeña en CPU. No cambia servicios activos.

El contrato JSON v1 tiene estas claves:

| Clave | Significado |
| --- | --- |
| `phase` | `development` o `final`, según la política de datos del repositorio. |
| `init_config` | Archivo con `network`, `training` y, opcionalmente, `outcome_width`. |
| `settings` | Presupuesto `actions`, `steps` por episodio (máximo 128), `seed`, intervalos de evaluación/checkpoint, episodios de evaluación, tasas de aprendizaje, congelación, replay y ajustes existentes del path planner. Los nombres y valores por defecto están en `SETTINGS` de `training_config.py`. |
| `thresholds` | Límites globales de calidad y requisitos de evidencia. |
| `curriculum` | Lista ordenada de `{source, game, levels, thresholds?, objective?}`. |
| `sdk` | Directorio de juegos, modo `offline`/`normal`, split de desarrollo y presupuesto `access_steps` para llegar al nivel. Solo necesario con entradas SDK. |

`levels: [3, 1, 2]` ejecuta ese orden. Primero se recorren los niveles de una
entrada y después la siguiente entrada. No se ordena ni se añaden niveles;
se rechazan pares juego/nivel duplicados y claves desconocidas. Solo se practica
y evalúa la etapa activa. Al cambiar de etapa se conservan pesos y optimizadores,
y se vacían los replays del predictor/planner para no practicar etapas anteriores
implícitamente. Cada etapa inicia sus propios permisos de aprendizaje.

Los umbrales pueden sobrescribirse dentro de cada entrada:

| Umbral | Condición |
| --- | --- |
| `representation_open` | Error máximo de reconstrucción para activar el predictor. |
| `prediction_open` | Error máximo de predicción a un paso y rollout de tres pasos. |
| `latent_drift` | Deriva máxima entre evaluaciones de las mismas observaciones. |
| `success` | Fracción mínima de niveles resueltos por el planner en etapas sintéticas. |
| `patience` | Evaluaciones consecutivas necesarias para cada permiso/avance. |
| `minimum_samples` | Transiciones mínimas en las evaluaciones de uno y tres pasos. |
| `minimum_changed` | Transiciones con cambios mínimas; la excepción estacionaria exige una terminación nativa observada. |

Los 71 sintéticos del catálogo permiten seleccionar directamente el nivel.
Square Cross y su variante admiten `objective: "subgoals"` (valor por defecto para
esos dos juegos): encoder/decoder, predictor y planner con su profesor.
Los 68 juegos de fundamentos/habilidades y Square Cross Maze (`mc01-v1`, niveles
1–6) usan `objective: "world_model"`:
representación y predictor con acciones aleatorias, sin un profesor compatible.
También se puede elegir ese modo para Square Cross. En el SDK,
no hay selección portátil de nivel ni profesor: se llega al nivel solicitado
mediante acciones aleatorias reales, hasta `sdk.access_steps`. Si no se alcanza,
el entrenamiento informa del error y no salta a otra etapa. El coste de acceso
se registra aparte de las acciones de entrenamiento; esos frames no entran al
encoder, replay ni pérdida. Tampoco entra la respuesta que cruza al siguiente
nivel. En las etapas `world_model` se entrenan representación y predictor con acciones
aleatorias; el avance requiere calidad predictiva, no éxito de un planner que
no se ha entrenado para esos juegos.

En desarrollo, `sdk.split` utiliza el formato de
[`configs/splits.example.json`](../configs/splits.example.json): juegos disjuntos
y límite de niveles por juego. El ejemplo ilustra el formato, no sustituye la
partición oficial completa de [la política de entrenamiento](../docs/training-policy.md).
Se rechaza entrenar un juego reservado o superar su límite y se reserva además
el último nivel informado por el SDK. `phase: final` permite los niveles públicos
reservados por desarrollo, siempre desde pesos aleatorios; nunca datos privados.
En todos los casos se usan observaciones/acciones públicas, sin leer reglas internas.

Para reanudar, conserva el archivo y la revisión del código:

```sh
.venv/bin/python -u -m model.subgoal_training \
  --training-config model/training_config.example.json \
  --output model/runs/mi-entrenamiento --device cuda --resume --actions 524288
```

La extensión opcional es un presupuesto **total** mayor. Se restauran pesos,
cursor, permisos y contadores; se reinician momentos del optimizador y replay
en RAM, igual que en la CLI anterior. Cambiar el plan/umbrales/arquitectura al
reanudar se rechaza. SIGUSR1 conserva su checkpoint atómico; SIGUSR2 avanza de
fase y SIGHUP avanza a la siguiente entrada del orden configurado. Un avance
manual no acredita los umbrales y deshabilita la publicación como run satisfactorio.

`budget_exhausted` significa que quedan etapas pendientes. `converged` indica
que se han superado sus gates; antes de publicar también se evalúan **todas** las
etapas con los pesos finales congelados y semillas distintas de desarrollo,
incluyendo éxito del planner en `subgoals`. Así se detecta el olvido de etapas
anteriores. La deriva temporal se acredita durante desarrollo; una evaluación
con pesos congelados no la inventa como cero. Los datos de evaluación no entrenan.

La preparación de una release verifica el plan, evidencia, checkpoint exacto,
ausencia de avances manuales, revisión de origen limpia y pesos finitos/cargables:

```sh
.venv/bin/python -m model.training_release \
  --run model/runs/mi-entrenamiento --output model/runs/mi-release
```

Produce `agent.pt`, `manifest.json` y `SHA256SUMS` en una carpeta nueva. Excluye
runtime, observaciones, replay, cachés y rutas privadas. No sube nada. El
[skill `arc3-training`](../skills/arc3-training/SKILL.md) describe la ejecución
desde cero y publicación autorizada como assets de GitHub Releases tras verificar
la descarga. Los pesos no se guardan como blobs en Git. Superar estas pruebas
locales no equivale a un resultado independiente de competición.

## Plan activo: aprendizaje incremental y currículo acumulativo

Sustituye las campañas anteriores. La campaña activa empieza desde cero: encoder,
decoder, predictor y PathValue con pesos nuevos; también contadores, criterios de
activación y resultados. Un solo juego (`sc01-v1`), hasta 262.144 acciones reales,
sin barrido de variantes. `subgoal_config.json` fija los tamaños y la receta
del modelo, sin depender de checkpoints previos para inicializarlo.

**Activación acumulativa de predictor y planner, con representación congelable:**

1. Por defecto, encoder y decoder aprenden desde la primera acción real y siguen
   durante toda la campaña salvo que se active la congelación al admitir el planner.
   El experimento opcional `--freeze-encoder` fija sólo
   el encoder y mantiene el aprendizaje del decoder.
   Reconstrucción de las dos últimas observaciones:
   CE de colores, delta visual y overlap, tanto con el decoder contextual como
   con su lectura local auxiliar (peso 1). La lectura local obliga a conservar
   información visual en z sin depender de la corrección contextual del decoder.
   SIGReg (peso 0,002) se usa sólo antes de la primera activación del predictor.
   Desde esa activación se apaga para todo el encoder compartido, incluso al
   abrir niveles nuevos o reanudar.
   En este arranque desde cero, AdamW LR 3e-5 mediante
   `--representation-learning-rate 3e-5`; se evalúan calidad y deriva antes de
   admitir el predictor. La campaña añade `--encoder-learning-rate-after-predictor 3e-6`:
   en la fase predictor, sólo el encoder baja a 3e-6; el decoder conserva 3e-5.
   Son grupos separados del mismo optimizador, sin reiniciar sus momentos al
   cambiar la tasa. Con la política por nivel, un nivel nuevo empieza con ambos a 3e-5.
   Las tasas son hipótesis de la campaña, no una garantía de convergencia.
   La campaña activa también usa `--freeze-representation-after-planner`: durante
   las partidas de un nivel con planner se congelan **encoder y decoder**.
   Al practicar un nivel nuevo vuelven a aprender, conservando los pesos.
   Los niveles anteriores siguen practicándose según su propia fase.
   Mientras están congelados, se detienen reconstrucción, regularizadores e invalidaciones del caché de z;
   las evaluaciones y vistas de reconstrucción siguen disponibles. Predictor y
   PathValue continúan aprendiendo. La memoria h aún se reconstruye con el predictor
   actualizado, y el replay sigue siendo secuencial; esto no elimina todo su coste.
2. El predictor se incorpora cuando la reconstrucción es ≤10 % y la deriva relativa
   del latente es ≤15 % en tres evaluaciones consecutivas del nivel. Desde entonces
   aprende **en cada acción válida**, aunque su error instantáneo sea pequeño.
   LR 3e-5; supervisión latente y visual a uno y tres pasos cuando hay historia.
   En el rollout, tanto z como h intermedios son predichos. Los targets son reales.
   La receta activa usa `prediction_delta_loss: "recall_focal_capped"`, con temperatura 1.
   Si `q` es la probabilidad del color real y `C` los píxeles realmente cambiados,
   suma `2*(-log(q))` en `C` y `lambda*(1-q)^2*(-log(q))` en los estáticos; divide
   por `max(32, |C|)` en cada imagen y promedia las imágenes del lote.
   `false_change_weight` fija lambda (1 en esta receta). La máscara y el divisor
   dependen de la realidad, sin argmax ni cortes de gradiente en los pesos suaves.
   Los estáticos bien conservados aportan poco, no cero; muchos estáticos dudosos
   pueden acumular una penalización grande. El suelo de 32 limita la amplificación
   en deltas pequeños; una imagen sin cambios conserva supervisión con divisor 32.
   No normaliza por el total de píxeles ni limita el valor máximo de la pérdida.
   El objetivo del predictor es **MSE latente + NLL visual**,
   tanto a un paso como en el extremo del rollout de tres pasos. No incluye
   overlap suave; `delta_overlap_weight` sigue aplicándose a la reconstrucción.
3. PathValue se incorpora cuando además los errores de predicción a uno y tres
   pasos son ≤10 % durante tres evaluaciones. Aprende al terminar cada segmento
   real de subobjetivo; el predictor continúa actualizándose. Encoder y decoder
   siguen la política de congelación por nivel descrita arriba.

Los permisos se mantienen una vez abiertos **para cada juego/nivel**. Una regresión
se muestra y bloquea avanzar de nivel, pero no apaga componentes ya habilitados.
La reconstrucción baja por sí sola no prueba estabilidad: la deriva compara z de
las mismas imágenes iniciales entre evaluaciones (norma RMS de la diferencia /
norma RMS anterior). La primera evaluación no tiene esa comparación y no abre
permisos. Los umbrales, paciencia y tasas son configurables; son valores iniciales,
no una conclusión experimental ni garantía de estabilidad.

La penalización adicional de varianza tiene peso **0** en esta campaña. Aplicarla
a sólo dos observaciones consecutivas exigía separar sus latentes en cada
posición, aunque la imagen apenas cambiase. SIGReg también usa una muestra pequeña;
conservarlo sólo al principio es una hipótesis de entrenamiento, no una garantía
contra el colapso. Los diagnósticos de diversidad, rango y respuesta de zonas
estáticas siguen activos. El cambio corrige además una omisión del entrenador
incremental: `local_reconstruction_weight` ahora participa realmente en la pérdida.
Los comandos de experimentos anteriores mantienen sus propias configuraciones.

**Un solo juego por campaña:** `--game sc01-v1` por defecto. `si01-v1` también
puede seleccionarse para otra campaña, pero no se intercalan ambos juegos.
Primero nivel 1; después 1+2; después 1+2+3, etc. Se alternan partidas nuevas
entre los niveles desbloqueados del juego elegido, con parámetros aleatorios. No se
borran pesos al avanzar; la experiencia temporal sí se descarta en cada partida.
El avance automático a otro nivel exige calidad de representación/predicción y ≥80 %
de éxito del planner en **el último nivel desbloqueado** de ese juego durante
tres evaluaciones consecutivas. No basta una media global ni agotar un presupuesto.

**Evaluación:** cada 256 acciones (`--evaluate-every`), al terminar la partida,
solo se comprueba el último nivel desbloqueado. Se juegan ocho partidas aleatorias
una vez por nivel y se conservan sus resultados como referencia fija. Las observaciones
y acciones reales se guardan atómicamente en `evaluation/` dentro del directorio
local de la campaña, separadas del replay de entrenamiento. Cada comprobación
recalcula representación y predicción con los pesos actuales sobre esas mismas
trayectorias, sin nuevas llamadas al entorno ni aprendizaje. No se guardan latentes
aprendidos: la calidad necesita recalcularse aunque el éxito aleatorio sea constante.
El planner se evalúa aparte una vez habilitado: ocho partidas con las mismas
semillas y presupuesto de 128 acciones. El historial de niveles anteriores sigue
visible, pero ya no detectamos regresiones periódicamente en ellos. El entrenamiento
acumulativo de niveles anteriores se mantiene.

La caché persiste al reanudar; un archivo incompatible o corrupto produce un error.
Las campañas antiguas sin trayectorias guardadas recogen la muestra una vez en la
primera comprobación de cada nivel. La migración guarda el protocolo nuevo antes
de entrenar, conserva pesos/contadores/permisos y reinicia las rachas de activación.
Las filas `quality_probe` no son nuevas partidas: tienen cero acciones nuevas y
resultados de victoria ausentes. La referencia `random` se registra una sola vez;
su línea permanece horizontal. La frecuencia sigue siendo 256 acciones, configurable.
La calidad predictiva requiere al menos ocho muestras y cuatro con cambios reales
a cada horizonte. Una predicción exacta sin cambios tiene error cero, pero no
sustituye la cobertura de transiciones con cambios. Para tareas que terminan sin
cambiar píxeles, se admite una evaluación enteramente estática solo si incluye
una terminación nativa observada; se decide por evidencia pública, no por el juego.
El éxito del planner sigue siendo obligatorio para avanzar automáticamente. La confirmación final utiliza
otras semillas y no modifica permisos ni currículo.

Calidad, deriva y activaciones usan la misma frecuencia. La deriva mide cambios
entre estas evaluaciones: acortar el intervalo cambia el criterio de estabilidad,
no solo la frecuencia del gráfico. Para cambiarlo al reanudar, se requiere
`--resume --evaluate-every 256 --update-evaluation-interval`. Se conservan pesos,
contadores y permisos ya abiertos, pero se reinician las rachas de comprobación.
La primera evaluación reconstruye los anclajes; las siguientes pueden contar
para las tres comprobaciones consecutivas. Cada fila registra su intervalo.

**Subobjetivos desde el inicio:** los juegos generan la lista después de
aleatorizarse. El agente recibe solamente la imagen/máscara del objetivo actual;
una lista de longitud uno usa el mismo contrato que una lista de varios objetivos.
Solo la confirmación real del profesor avanza la lista. La memoria recurrente se
conserva entre objetivos de la misma partida. El mínimo de acciones es una referencia
para métricas, nunca una ruta ni una etiqueta privilegiada para decidir acciones.

**Path Planner:** búsqueda latente con horizonte 3, beam 4 y 16 candidatos uniformes
de clic. Replanifica tras cada acción real; cada rama imaginada conserva su propio h.
Tras habilitarse, exploración de 100 % a 20 % en sus primeras 8.192 acciones por
nivel por defecto. `--path-exploration-warmup 2048` acorta ese arranque; conserva
el mínimo del 20 % y cuenta sólo acciones del planner del propio nivel. Antes de
habilitarlo se mantiene la recogida aleatoria. El valor se guarda y se hereda al
reanudar; un cambio explícito conserva pesos, contadores, permisos y rachas de
evaluación, y se registra antes de recoger más experiencia. Las evaluaciones
fijas siguen sin exploración del planner. PathValue usa features compartidas y el presupuesto restante para aprender
`P(éxito dentro del presupuesto)` con BCE y `log1p(acciones restantes | éxito)`
con SmoothL1, sólo para segmentos exitosos (cero en el estado final alcanzado).
Se entrena con cuatro pasadas y LR 1e-4 sobre una mezcla de hasta ocho segmentos
reales completos, conservados en un replay de hasta 64 segmentos y limitado por
el presupuesto de bytes del motor. Cada segmento tiene el mismo peso, aunque su
partida sea más larga. Se guardan observaciones y objetivos originales, y se
recalculan sus latentes y memorias con los pesos actuales. El fracaso
observado supervisa la probabilidad; no significa que ese estado sea irresoluble.
El presupuesto de cada ejemplo es el de la partida menos las acciones ya gastadas,
también al cambiar de subobjetivo. No se renueva artificialmente.

Tanto la poda del beam como la elección final priorizan probabilidad: entre rutas
a menos de 0,02 del máximo, eligen menor duración total (prefijo imaginado más
restante condicionada al éxito). Los empates numéricos de duración (1e-6 acciones)
se aleatorizan con el RNG de la partida; no se prefiere A1 por orden de enumeración.
La probabilidad procede de la política de recogida y puede estar mal calibrada o
explotar errores del predictor. No sustituye la confirmación real del profesor.

`model.path_tuning` compara márgenes sobre un checkpoint fijo y las mismas semillas
de desarrollo, sin entrenar ni generar pesos. Por ejemplo, tras copiar el checkpoint
a una ruta estable fuera de Git:

```sh
.venv/bin/python -m model.path_tuning --source /tmp/planner-fixed.pt \
  --output /tmp/planner-margin-screen --levels 1 2 --margins .02 .05 .10 \
  --episodes 16 --dev-offset 710000
```

El primer margen es la referencia. `--horizons 3 2` compara también profundidad de
búsqueda sin modificar los horizontes de entrenamiento; la primera combinación es
la referencia de cada nivel. `--epsilons .2 .1` compara exploración aleatoria con
los mismos pesos; por defecto es cero como en la evaluación fija del planner.
`--noop-cooldowns 0 8` compara la exclusión temporal de acciones observadas sin
efecto. `--leaf-only` es una variante experimental que puntúa sólo rutas de toda
la profundidad; está desactivada en la campaña. Ninguna opción cambia pesos.
El informe conserva semillas, hash del checkpoint,
victorias ganadas/perdidas, acciones totales, acciones de partidas exitosas y acciones
en las mismas partidas ganadas por ambos candidatos. Reducir sólo la media de los
éxitos puede ocultar que ahora fracasan las partidas largas. Un candidato debe
confirmarse frente a la referencia en otro conjunto de desarrollo (`--dev-offset`
distinto); estas comparaciones no demuestran equivalencia estadística ni cambian
el currículo o la configuración del entrenamiento activo.

En la comparación del checkpoint de la acción 126984 no se adoptó ningún ajuste
global. Ampliar el margen de .02 a .03 sumó una victoria en cada nivel de la muestra
de confirmación, pero no acortó ninguna de las rutas exitosas compartidas. Un
horizonte de dos perdió una victoria del nivel 1 en el cribado. Bajar exploración
de .20 a .10 redujo acciones en algunos éxitos, pero en la confirmación del nivel 1
bajó de 21/32 a 17/32 victorias (nivel 2: 22/32 con ambas). Se conservaron esos
tres parámetros. Tras activarse manualmente el planner del nivel 2, su exploración
de arranque estaba al 91 %. Una comparación posterior con pesos de la acción
129057 confirmó que bajarla de .90 a .65 pasó de 13/32 a 16/32 victorias; en las
diez partidas ganadas por ambos bajó de 139 a 94 acciones. Se adoptó un arranque
de 2.048 acciones, manteniendo el mínimo .20: los niveles que ya completaron el
arranque permanecen igual. Es evidencia con pesos fijos, no una garantía de la
evolución futura del aprendizaje. Los resultados completos por semilla están en
[`path_tuning_evidence.json`](path_tuning_evidence.json).

En el checkpoint 140330, ampliar de tres a seis pasos no cambió las victorias ni
las acciones en 16 partidas por nivel y multiplicó por aproximadamente 2,2 el coste de decisión.
Puntuar sólo las hojas empeoró los resultados. Se conserva profundidad 3 y se
adopta `--path-noop-cooldown 8`: dos muestras separadas de 32 y 64 partidas pasan,
en total, de 61/96 a 64/96 victorias en nivel 2 y conservan 71/96 en nivel 1.
La segunda muestra intercambia una victoria por una derrota en nivel 1: no hay
garantía por partida ni significación estadística demostrada. Los bloqueos de
2/4 acciones no mejoran el resultado global de la muestra de 64. Se rechaza usar
la probabilidad de éxito futuro como señal de WIN actual: tampoco se separan de
forma fiable los estados predichos exitosos mediante distancia al objetivo.
No se habilita poda por éxito imaginado ni reutilización no verificada de rutas.
La evidencia y semillas están en [`path_budget_evidence.json`](path_budget_evidence.json).
Ese archivo incluye el nuevo divisor del loss sobre las 211 transiciones reales
ya analizadas: media visual 5,109 → 2,299 y máximo 114,329 → 17,864.
Es un cambio de escala, no una mejora demostrada de precisión.

Reward mostrado: `-(acciones - mínimo inicial) - 128 * fracaso`; se conserva como
métrica histórica, no como criterio de selección ni target de PathValue.
Un corte administrativo queda censurado, no se etiqueta como fracaso.
La versión 3 normaliza las features antes de las dos salidas con LayerNorm; la
versión 4 además usa contraste compartido entre estado y objetivo dentro de la
máscara, para que el fondo no domine los pequeños cambios relevantes. La versión
4 es el valor inicial para evaluadores nuevos. La versión se almacena en el
checkpoint: cargar una versión anterior no cambia su cálculo.

`--path-hindsight` añade un objetivo realmente alcanzado dentro de cada segmento
muestreado, hasta su primera aparición; conserva el fracaso del objetivo original
y excluye imágenes de GAME_OVER. `--path-predicted-inputs` empareja los estados
reales con predicciones de uno a tres pasos usando las acciones que efectivamente
se ejecutaron. Ambos reciben las mismas etiquetas reales: las predicciones nunca
aportan objetivos de éxito ni gradientes al modelo del mundo. Las dos opciones
son explícitas y se heredan al reanudar; por defecto están desactivadas.

**Coherencia de los pesos:** representación, predictor y planner tienen optimizadores
separados sobre los mismos módulos del agente. Tras actualizar representación se
invalidan los latentes de inferencia, se recodifican historial y objetivo, y se
recompone h. El predictor compara ambos extremos con el encoder actualizado y usa
el decoder sin actualizar sus pesos en ESA llamada; el decoder sí aprende con su
propio optimizador. Después de modificar dinámica se recompone h otra vez. No
actualizamos pesos dentro de una búsqueda ni mezclamos predicciones con datos reales.

CUDA usa `torch.compile`, BF16 y momentos AdamW de 8 bits para representación y
predictor; CPU usa float32 y AdamW normal. Se conserva el límite de GPU a 250 W.
Por defecto, la partida actual es el único buffer. El experimento opcional
`--predictor-replay-episodes 16 --predictor-batch-size 4` mantiene hasta 16 partidas
reales de entrenamiento en RAM, con el límite de bytes de replay del modelo.
Cada actualización mezcla la transición recién observada con un extremo aleatorio
de hasta tres partidas anteriores distintas; al arrancar usa las disponibles.
Conserva el historial real de cada muestra y reconstruye su memoria recurrente con
los pesos actuales. Los latentes cacheados se invalidan si cambia el encoder.
Se promedia la misma pérdida por transición, incluido el rollout de tres pasos
cuando existe historial suficiente. Se acumulan gradientes por muestra y se hace
un solo paso del optimizador por acción real, con clipping después de promediar.
El término de rollout ausente aporta cero a esa media; si falta en todo el lote,
no se registra. Las evaluaciones nunca entran en el buffer. No se guardan
grabaciones/replay en disco: al reanudar el buffer vuelve a estar vacío.
Los dos parámetros se heredan del checkpoint, o pueden cambiarse explícitamente
con `--resume`; el cambio conserva pesos, contadores y permisos, reinicia las
rachas y se guarda antes de entrenar. Para desactivarlo se usan capacidad 0 y
lote 1. El dashboard muestra la ocupación y el tamaño real del último minibatch.
Checkpoint portátil inicial y cada 2.048 acciones, al terminar la partida. Incluye
pesos y configuración de módulos activos y resumen del currículo. Este comando
inicia una campaña nueva. Con `--resume` continúa desde `latest.pt` del directorio
de salida: recupera todos los pesos (incluido PathValue), permisos, niveles y
contadores, y descarta mediciones posteriores al checkpoint. Los momentos de los
optimizadores se reinician: no es una continuación numéricamente idéntica. También
se renuevan los anclajes de deriva en la siguiente evaluación. No cambia los
hiperparámetros ni el juego al reanudar; permite ampliar el presupuesto de acciones.
La excepción explícita `--update-representation-threshold --representation-error 0.10`
permite cambiar únicamente ese umbral con `--resume`.
`--update-latent-drift-threshold --latent-drift 0.15` permite cambiar el límite de
deriva. Ambas opciones conservan pesos, contadores y permisos abiertos, pero
reinician las rachas de evaluaciones si cambia el umbral para exigir evidencia nueva.
`--prediction-delta-loss recall_focal_capped` permite seleccionar explícitamente la nueva NLL,
también con `--resume`. Sin esta opción se hereda la pérdida del checkpoint o del
JSON de inicialización; los checkpoints anteriores siguen usando `balanced`.
Los modos anteriores `balanced`, `union`, `focal` y `recall_focal` siguen disponibles sin cambiar sus fórmulas.
El cambio conserva pesos, permisos y contadores, reinicia las rachas y guarda
el nuevo modo antes de continuar. El reporte registra el cambio y su acción.
La reconstrucción del encoder/decoder conserva su pérdida anterior.
`--path-horizon N` cambia sólo la profundidad de búsqueda; el entrenamiento y la
evaluación del predictor conservan tres pasos. `--path-noop-cooldown N` activa una
exclusión temporal en la raíz: tras una respuesta real sin cambios dentro de la
máscara del objetivo, evita la misma acción/datos desde ese mismo tablero durante
N acciones reales, siempre que queden alternativas. No altera las acciones legales
ni poda estados imaginados. Caduca, se borra al cambiar de objetivo/episodio y
permite todas las acciones si no quedan alternativas. Es una heurística falible
con observación parcial: una acción aparentemente ineficaz podría tener efectos
ocultos o retardados. Cero conserva el comportamiento anterior. Ambas opciones
se heredan del checkpoint al reanudar; un cambio explícito conserva pesos y
permisos, reinicia rachas y guarda la configuración antes de actuar.
Para convertir un PathValue antiguo se exige `--migrate-path-value`: conserva sus
features compartidas y todos los pesos del mundo, pero reinicia las dos salidas
(éxito inicial 0,5; duración inicial 16 acciones) y añade la entrada de presupuesto.
La migración desde versiones 1/2 produce `objective_version: 3`, con LayerNorm
antes de las salidas. Los siguientes checkpoints no necesitan migración. La
versión 2 sigue siendo cargable sin migración para controles congelados; la
versión 4 de contraste se inicializa con pesos nuevos; no se reinterpretan
silenciosamente las features de un checkpoint anterior.
Se mantienen contadores/permisos, se reinician rachas de promoción y se guarda antes
de actuar. Como en todo resume, los optimizadores y el replay RAM se reinician.
El reporte registra las transiciones de objetivo; pérdidas anteriores y nuevas no
son directamente comparables. La trayectoria muestra por separado la estimación
de la elección y la de la ruta visualizada, que puede ser una alternativa.

`python -m model.path_recovery --source CHECKPOINT --output DIRECTORIO_NUEVO`
permite aislar el aprendizaje del evaluador con el mundo congelado. Usa semillas
de entrenamiento y desarrollo disjuntas, compara con azar y comprueba que la
evaluación no cambie pesos. `--fresh-path-value --path-objective-version 4`
inicializa el evaluador de contraste; `--hindsight --predicted-inputs` activa sus
dos aumentos. `--policy planner --epsilon .25` recoge partidas de la política
actual con exploración; por defecto recoge acciones aleatorias. `--episodes 0`
solo evalúa. El reporte contabiliza por separado todas las partidas, acciones y
actualizaciones de esta recuperación. Su checkpoint no sustituye directamente
al del currículo: no contiene los contadores ni permisos de la campaña original.

La recuperación del 16/10/2026 en `sc01-v1`, nivel 1, pasó de 7/32 a 25/32
victorias en el mismo conjunto adicional de desarrollo (azar: 10/32), con el
mundo congelado durante el diagnóstico y entrenamiento del evaluador. Supuso
1.536 partidas reales adicionales y 29.084 acciones, separadas de las curvas de
la campaña. Las evaluaciones de ocho partidas siguen oscilando; no son resultados
de competición ni prueban descubrimiento autónomo de objetivos. Semillas,
contadores y huellas de los checkpoints: [evidencia](path_recovery_evidence.json).
Tras reanudar la campaña con el predictor aprendiendo, las primeras 88 partidas
terminadas dieron 67 victorias y las tres evaluaciones fijas fueron 5/8, 6/8 y
5/8. Encoder y decoder conservaron sus pesos durante esa verificación.
`--encoder-learning-rate-after-predictor 3e-6` también admite un cambio explícito
al reanudar. El valor debe ser positivo y no mayor que el LR de representación.
Se guarda en el checkpoint y se hereda al omitir la opción; checkpoints antiguos
sin ella conservan el LR anterior. El cambio preserva pesos, contadores y permisos,
reinicia las rachas de cualificación y se guarda antes de entrenar. El dashboard
muestra las tasas efectivas; un encoder congelado sigue excluido del optimizador.
Los momentos de Adam y el replay en RAM se reinician al reabrir el proceso, como
en las demás reanudaciones; reducir la tasa dentro del proceso no los reinicia.
`--freeze-representation-after-planner` se guarda y hereda desde el checkpoint;
los checkpoints antiguos sin la opción mantienen su comportamiento anterior.
La fase se selecciona para cada partida: representación, predictor o planner,
según los permisos de ese nivel. Se restaura el nivel activo al reanudar.
Los cambios de fase dentro del proceso conservan los momentos del optimizador.
Sin esta política, la reducción de LR del encoder sigue siendo global tras
la primera admisión del predictor, como en las versiones anteriores. `--no-freeze-representation-after-planner`
al reanudar la desactiva explícitamente, respetando después las opciones de LR
y congelación sólo del encoder. Las pérdidas de reconstrucción no calculadas
se registran como ausentes, no como ceros.
`--freeze-encoder` fija sólo el encoder: el decoder y el predictor siguen
aprendiendo con las mismas pérdidas. No se optimizan ni regularizan los pesos
del encoder. El modo queda en el checkpoint y se hereda con `--resume`; los
checkpoints antiguos mantienen el encoder entrenable. `--no-freeze-encoder`
lo vuelve a activar. Cambiar este modo al reanudar conserva pesos, contadores y
permisos, reinicia las rachas de evaluación y guarda antes de seguir entrenando.
El dashboard identifica el encoder congelado; el contador de representación
pasa a contar actualizaciones del decoder. La primera evaluación tras reanudar
establece nuevos anclajes de deriva, por lo que no se compara con la anterior.
Este experimento no cambia la NLL ni demuestra por sí solo una mejora predictiva;
también se reinician los momentos de los optimizadores al reanudar.
`--init-config` inicializa todos los módulos aleatoriamente con la semilla indicada.
La alternativa explícita `--source` transfiere el mundo preentrenado e inicializa
PathValue; no se utiliza en la campaña actual. El dashboard identifica ambos modos.

En Linux, `SIGUSR1` solicita un checkpoint adicional y el entrenamiento continúa.
La señal sólo encola la petición: los pesos y contadores se guardan atómicamente
en `latest.pt` al terminar la partida o la evaluación en curso. Varias señales
pendientes se agrupan en un guardado; una recibida durante la escritura queda
pendiente para el siguiente punto seguro. Para el servicio de esta campaña:

```sh
systemctl --user kill --kill-whom=main --signal=SIGUSR1 arc3-curriculum.service
```

Para un proceso lanzado desde un shell, usa `kill -USR1 <PID>`. El log confirma
el guardado con `checkpoint_saved`, `reason: "signal"` y el contador de acciones;
`report.json` actualiza `checkpoint_action` y `last_checkpoint`. El campo
`checkpoint_signal: "SIGUSR1"` indica que esta versión tiene instalado el handler.
También hay dos controles manuales (se aplican al terminar la partida o evaluación):

```sh
# Una fase: representación -> predictor -> planner, en el nivel de entrenamiento actual.
systemctl --user kill --kill-whom=main --signal=SIGUSR2 arc3-curriculum.service
# Abrir el siguiente nivel y practicarlo en la próxima partida.
systemctl --user kill --kill-whom=main --signal=SIGHUP arc3-curriculum.service
```

Durante una evaluación, la fase se refiere al último nivel de **entrenamiento**,
no al nivel que se está evaluando. Un avance de nivel seguido de otro de fase
actúa sobre el nivel recién abierto. Se conservan las prácticas de niveles previos.
Estas peticiones saltan los gates explícitamente: no inventan mediciones ni rachas
aprobadas. Se registran en `manual_controls` (últimas 128), en el dashboard y en
el checkpoint antes de la siguiente acción; `control_signals` indica disponibilidad.
La señal sólo encola; si llega durante una escritura se atiende en el siguiente
punto seguro. Las señales POSIX pueden agruparse: espera la confirmación antes de
repetir. Si ya está en planner, en el último nivel o sin presupuesto, se registra
sin avanzar. Una escritura fallida detiene el proceso y conserva el checkpoint anterior.
No enviar estas señales a procesos antiguos que no tengan los handlers. Se mantiene el formato
portátil descrito arriba: guarda pesos y progreso, sin momentos del optimizador.

```sh
.venv/bin/python -u -m model.subgoal_training \
  --game sc01-v1 \
  --init-config model/subgoal_config.json \
  --representation-learning-rate 3e-5 \
  --encoder-learning-rate-after-predictor 3e-6 \
  --freeze-representation-after-planner \
  --output model/runs/curriculum/current
.venv/bin/python -m model.dashboard \
  --subgoal-report model/runs/curriculum/current/report.json
```

Para habilitar **Avanzar fase** y **Siguiente nivel** en la pestaña Currículo,
arranca el dashboard con `--curriculum-service arc3-curriculum.service` además de
`--subgoal-report`. Sin esa opción continúa siendo de sólo lectura. Los botones
usan los controles del entrenador activo, muestran la petición pendiente y el
resultado registrado; se desactivan mientras esperan o cuando el avance no procede.
Se comprueba que el servicio corresponde al reporte mostrado. No reinician el
entrenamiento ni cambian pesos desde el servidor web. El backend requiere Linux
con pidfd y el mismo usuario del servicio. Sólo hay que reiniciar el dashboard
para habilitarlos; no es necesario reiniciar el entrenador que ya admite señales.

LMEX sigue `model/runs/curriculum/current/latest.pt`; el sincronizador reutiliza el
`architecture.json` del repositorio. El checkpoint `path_mpc` necesita que su
llamador proporcione el objetivo con `Agent.set_goal`: no descubre objetivos.
El profesor sintético está fuera del agente y se rechaza para juegos SDK ajenos
al repositorio. Los resultados no prueban descubrimiento autónomo de reglas/objetivos.

**Dashboard:** una ejecución de un solo juego, una tabla de niveles, componentes habilitados,
mediciones y evaluaciones consecutivas para avanzar. La cabecera muestra el uso
de GPU (%) y su temperatura (°C), consultados cada dos segundos con `nvidia-smi`
desde el servidor HTTP. Son lecturas actuales de toda la GPU, incluidos otros
procesos; no se guardan históricos ni se modifica el entrenador. Si no hay lectura,
se indica como no disponible, sin conservar un valor antiguo ni inventar un cero.
Curvas fijas de reconstrucción
y predicción a uno/tres pasos, agregadas con igual peso por nivel; se dejan cortes
al cambiar la cobertura del currículo. Cada leyenda muestra el último valor
medido, con su acción en la ayuda. Al pasar el cursor, tocar o enfocar una leyenda,
se destaca su curva y aparece una guía horizontal punteada en el último valor
medido, con un círculo en su posición. La selección persiste entre consultas;
no se inventa una referencia para series sin datos. La gráfica de calidad marca los umbrales
configurados con líneas finas punteadas; los límites coincidentes comparten línea.
Son referencias: el avance requiere cumplir todos los criterios por nivel, no
solo que una media cruce una línea. Las curvas de interacción conservan las
últimas 4.096 acciones: bloques de 32 y tendencia de 512. El error suma errores y
uniones de deltas; el desglose usa píxeles por acción medida. La pérdida de
representación y las métricas del planner también muestran medias
móviles de hasta 512 acciones, con los valores individuales atenuados. Las pérdidas
solo cuentan actualizaciones registradas; la eficiencia promedia por objetivo
alcanzado y el reward incluye fracasos. Los subobjetivos se sitúan en la acción
de cierre de su partida, cuando se publican. Datos ausentes no son
ceros, y bloques sin cambios se marcan aparte. Las evaluaciones tienen sus propios
contadores, nunca suman acciones de aprendizaje.

Junto a **Eficiencia por subobjetivo**, **Resultado por subobjetivo** muestra los
éxitos y fracasos reales de entrenamiento como puntos en 100/0 y su porcentaje de
éxito en una ventana de 512 acciones. Los detalles indican juego, nivel, acciones
gastadas y mínimo inicial; el resumen muestra éxitos, fracasos y denominador.
Se agrupan todos los subobjetivos publicados en la misma acción antes de calcular
el porcentaje. Cada segmento terminado cuenta una vez, incluidos éxitos de cero
acciones; registros sin resultado booleano o reward terminal quedan fuera. Esta
curva incluye la exploración y los niveles registrados: no representa éxito
imaginado ni sustituye las evaluaciones fijas por nivel del planner. La eficiencia
sigue midiendo únicamente subobjetivos alcanzados.

La cabecera y la gráfica de velocidad muestran acciones de entrenamiento por
segundo en los **últimos 60 segundos** de reloj (ventana menor al arrancar).
Incluyen el tiempo de evaluación, guardado y pausas; excluyen la compilación
inicial anterior a la primera acción. El contador se reinicia al reanudar el
proceso y llega a cero tras 60 segundos sin acciones. El campo `recent_tps`
identifica esta métrica: el dashboard no reutiliza como recientes los valores
acumulados de informes antiguos.

Para atribuir costes sin alterar la campaña, `model.profile_training` copia
`latest.pt`, el informe y las grabaciones de evaluación a un directorio nuevo.
La copia sigue las reglas normales de reanudación: mismos pesos/progreso, pero
optimizadores y replay de entrenamiento nuevos. Pausa antes las cargas que
compitan por la GPU. Tras `--`, pasa los argumentos originales del entrenador
excepto `--output` y `--actions`; por ejemplo, para la configuración anterior:

```sh
.venv/bin/python -m model.profile_training \
  --source-run model/runs/curriculum/current \
  --output model/runs/profile/example --actions 256 --warmup 64 -- \
  --game sc01-v1 --init-config model/subgoal_config.json \
  --representation-learning-rate 3e-5 \
  --encoder-learning-rate-after-predictor 3e-6 \
  --freeze-representation-after-planner
```

`timing.json` separa las categorías por tiempo de pared exclusivo, incluidas sus
esperas. La ventana empieza después del calentamiento y termina al completar el
presupuesto; incluye evaluaciones periódicas, pero excluye la evaluación final
posterior. Compara mismos pesos, semillas y opciones, comprobando que no queden
compilaciones tardías. `--cpu-profile` guarda `cpu.prof`; `--trace-actions 8`
captura CPU/kernels CUDA antes de la ventana, y `--synchronize` espera en los
límites de las categorías externas. Estas opciones alteran los tiempos: confirma
las ganancias sin ellas y no sumes anotaciones GPU anidadas como utilización.
También admite `--device cpu` entre los argumentos del entrenador.

El entrenamiento publica JSON compacto con el mismo contenido y frecuencia.
La memoria del replay se reconstruye con pesos actuales, transfiriendo sus
acciones juntas; se omite una reconstrucción previa que ese replay no utilizaba.
Las pérdidas del predictor se copian a CPU una vez por lote, sin cambiar sus
gradientes, pesos relativos ni la acumulación numérica de las métricas.

**Subobjetivos y Path Planner → Éxito del planner** compara el porcentaje de
niveles resueltos (`100 × victorias / partidas`) del planner y del azar en las
evaluaciones fijas de desarrollo. El selector separa cada juego/nivel; el eje
horizontal cuenta acciones de entrenamiento. Cuando hay mediciones del planner,
la gráfica empieza en la primera para que su evolución sea legible. Las leyendas
y los puntos muestran los denominadores y la acción de cada medición; un cero
medido se conserva y una evaluación ausente no se rellena. Las evaluaciones
finales y las partidas de entrenamiento no se mezclan en estas curvas. La línea
del umbral de éxito no sustituye los demás requisitos de promoción del currículo.

**Subobjetivos y Path Planner → Trayectoria real e imaginación** aparece debajo
de la observación y el subobjetivo de la muestra seleccionada. El tiempo avanza
hacia la derecha: la fila real contiene el estado final de cada acción y su
etiqueta; cada fila inferior parte de un origen `t` y alinea sus predicciones
en las columnas `t+1`, `t+2`, `t+3`. Son reconstrucciones de los latentes de una
ruta explorada hasta el horizonte disponible, capturados antes de ejecutar la
acción o aprender de su resultado. La ruta que decide la acción puede ser más
corta: el visor prioriza una continuación de ese prefijo dentro de las ramas
que sobrevivieron a la búsqueda. Si no queda ninguna, prioriza la misma primera
acción y, en último caso, muestra la mejor alternativa de ese horizonte. Las
filas distinguen «Ruta elegida», «Continuación explorada» y «Alternativa explorada»;
esta selección para inspección no cambia la decisión ni las puntuaciones.
Se decodifican hasta tres estados ya calculados, sin repetir la
búsqueda ni ejecutar nuevas predicciones, sin gradientes y sin cambiar pesos.
La captura añade ese coste de lectura/decodificación sólo a las partidas elegibles.
El presupuesto restante puede acortar el horizonte; no se rellena. Las secuencias que difieren de las acciones
finalmente ejecutadas se marcan «Ruta distinta»; no son medidas de error de
predicción sobre la trayectoria real. Los destinos posteriores al final real
se muestran sin observación de referencia.

Seleccionar un estado fija la muestra y sincroniza las dos tarjetas superiores.
La reproducción y la barra recorren además todos los frames de animación reales.
Se distinguen entrenamiento/evaluación y exploración aleatoria; esta última no
reutiliza rutas anteriores del planner. Las evaluaciones de la política aleatoria
no se capturan. La casilla de actualización automática permite recibir otra
muestra, sin reconstruir miniaturas cuando su identificador no cambia. Las
muestras antiguas siguen funcionando con avisos de subobjetivos/predicciones
ausentes. Se conserva una sola trayectoria completa en
`planner_trajectory.json`, sustituida como máximo cada dos segundos al finalizar
una partida, sin guardar un archivo histórico ni añadir datos al checkpoint.
El reporte incluye solo su identificador/contexto y `/api/trajectory` sirve
exclusivamente la muestra que coincide con ese identificador. Los frames se
guardan como bytes de paleta en base64; una trayectoria que exceda 4 MiB entre
frames reales, predicciones y subobjetivos, o 128 acciones, se omite entera con
un aviso, sin truncarla. Un error de captura o escritura conserva la muestra
anterior y no detiene el entrenamiento. Al
reanudar se descarta la referencia a la muestra del proceso anterior.

Cada gráfica de pérdida permite desplegar los componentes de la última
actualización, con sus pesos ya aplicados: su suma coincide con el total. La
representación distingue CE, delta, overlap, reconstrucción local y regularizadores
cuando se calculan. El predictor distingue MSE latente, delta visual y rollout
cuando existe un prefijo suficiente. Un término ausente no se presenta como cero.
El panel muestra también si SIGReg sigue activo en la fase inicial o está apagado.
La gráfica **Pérdida por componente** muestra además el histórico del predictor:
MSE latente, NLL visual y total del rollout. Los registros anteriores conservan
el error de solapamiento que sí usaron; las nuevas actualizaciones lo omiten.
Reutiliza `curves[].predictor_terms` de `report.json`; esos términos también están presentes
en los eventos del log. Cada curva conserva las actualizaciones individuales y
su media sobre hasta 512 acciones, contando sólo valores calculados. El rollout
no existe en todas las actualizaciones, por lo que las medias por componente
pueden no sumar la media de la pérdida total. Se indica el modo activo de NLL;
un reporte antiguo sin ese campo no recibe un modo inventado. Estas pérdidas
no son porcentajes de acierto ni recompensas; los aciertos aparecen en el desglose
de deltas. No hace falta reiniciar el entrenamiento para mostrar este histórico.

El panel del predictor muestra cada dos segundos una muestra aleatoria de las
últimas 64 transiciones medidas: tablero previo real, reconstrucción del previo,
delta real, delta predicho y una máscara de su unión. La máscara clasifica cada
píxel cambiado como inventado, omitido, color incorrecto o acertado, con los mismos
colores del desglose y un recuento por categoría; los píxeles sin cambios en ambos
deltas conservan la cuadrícula. La reconstrucción decodifica el latente de esa misma
observación, con los pesos anteriores a la actualización: durante el aprendizaje
reutiliza la salida del decoder ya calculada; en evaluación se decodifica sin
gradientes. No es una visualización directa de los valores del vector latente. Se
reutiliza la predicción causal obtenida antes de observar el resultado y actualizar
pesos; no se vuelve a inferir para dibujarla. Ambos deltas comparan el resultado
final de la acción con el mismo tablero previo, con código 16/cuadrícula para
«sin cambios». La muestra identifica juego, nivel, partida, acción y fase; no
confunde evaluación con entrenamiento ni convierte ausencia de cambios en error
cero. Este pequeño buffer visual es efímero, usa un RNG independiente y no es replay
de aprendizaje. El informe conserva únicamente la muestra que se está mostrando.

El diagnóstico del latente se registra en `quality.latent` durante la misma
evaluación aleatoria fija, sin optimizar ni hacer nuevas pasadas por el encoder.
Reutiliza los estados reales con pesos congelados; no consume RNG de entrenamiento
ni incorpora trayectorias imaginadas. No cambia las activaciones del currículo.

- **Diversidad** (`std_mean`): desviación típica poblacional entre observaciones,
  promediada sobre las coordenadas del latente. Compara las mismas posiciones;
  nunca mezcla geometrías para aparentar variación. `within_episode_std` mide
  lo mismo dentro de cada partida antes de agregar, para distinguir cambios
  temporales de diferencias de fondo/color entre escenarios.
- **Coordenadas casi constantes** (`inactive_fraction`): fracción con desviación
  menor que 0,01. Es un umbral diagnóstico en unidades del latente, no una puerta
  de entrenamiento ni una garantía universal de colapso.
- **Rango efectivo** (`effective_rank`): `exp(-sum(p * log(p)))`, con `p` igual a
  los autovalores normalizados de la covarianza de los vectores centrados. Se
  calcula mediante la matriz de Gram, sin normalizar/reescalar los latentes.
  El máximo `max_rank` es `min(observaciones - 1, coordenadas)`. Un latente
  exactamente constante tiene rango cero por convenio; menos de dos observaciones
  deja el diagnóstico sin datos. Un rango bajo también puede reflejar una tarea
  simple o pocas observaciones; no es el porcentaje de capacidad usado.
- **Respuesta a cambios reales** (`changed_rms`, `unchanged_rms`): RMS de
  `z_después - z_antes`, separando bloques que contienen algún píxel realmente
  cambiado y bloques estáticos. Usa todas las transiciones evaluables de la partida,
  con los mismos pesos para ambos estados. Respeta el tamaño de patch del encoder
  y los bordes parciales. El contexto puede cambiar un bloque estático: esta medida
  no es un error de predicción. Sin bloques de una clase, su valor es `null`.

Para acotar el coste, diversidad/rango usan las primeras ocho observaciones reales
por partida, incluida la inicial, guardadas temporalmente en CPU; el rango usa
como máximo 64 por geometría, seleccionadas a intervalos deterministas. Las medias
por geometría ponderan observaciones, y el dashboard agrega niveles con igual peso.
Los recuentos de observaciones, partidas, geometrías, transiciones y bloques se
registran junto a los valores. Los tensores de diagnóstico no se guardan en pesos
ni informes. Las curvas empiezan cuando existe medición y se cortan al cambiar la
cobertura del currículo; los informes anteriores mantienen sus datos ausentes.

Quedan fuera: Goal Planner aprendido, actor por imitación, Double-DQN, barrido de
arquitecturas y etiquetas de distancia privilegiadas para PathValue. Se observa
primero si las activaciones progresivas y el repaso de niveles estabilizan el
aprendizaje, antes de introducir otra variante.

Validación proporcional: contratos CPU de activación acumulativa, cobertura,
promoción, ownership de gradientes, renovación de latentes y rollout autónomo;
contratos existentes de múltiples subobjetivos y API SDK; checkpoint del mismo
agente; HTTP/JS del dashboard; prueba corta CUDA compilada. Sin pruebas visuales.

## Protocolos anteriores — fuera de la campaña activa

Los apartados de comparaciones y probes que siguen documentan comandos existentes.
Sus resultados y criterios de promoción no se aplican al plan activo anterior.

El objetivo de esta iteración es **entrenar, guardar y ejecutar el mismo agente**,
con `z_next = z + delta` y planificación en latentes. El planner conserva la
decisión final; un actor aprendido propone acciones y secuencias. Su criterio
principal es progreso SDK esperado más valor futuro, nunca similitud visual
sumada como si fuera probabilidad de éxito. El primer progreso y GAME_OVER son
absorbentes para el objetivo local. Un éxito en la última acción permitida sigue
teniendo target uno; el presupuesto agotado elimina únicamente el futuro.

La ejecución se divide en estas entregas, manteniendo una referencia de pesos
inmutable y sin convertir cada hipótesis en una pieza obligatoria:

1. Agente compartido, checkpoint conjunto y equivalencia entre actuación local
   y adaptador de despliegue. Entrenador separado, mismos módulos; evaluación
   congelada y sin replay. La adaptación online no está habilitada implícitamente.
2. Presupuesto explícito en V/Q; targets MC y Double-DQN sobre sucesores reales.
   Q online selecciona la acción y Q target evalúa esa misma acción; EMA después
   del optimizador. Q de clics usa el mismo retorno en la coordenada ejecutada.
3. Entrenamiento de uno más dos pasos autorregresivos, con targets reales y
   máscaras de fronteras. `loss_steps` cuenta endpoints, no horizonte imaginado.
   Comparación congelada a uno/dos/cuatro pasos, incluida predicción identidad.
4. Planner de progreso/valor con horizonte inicial dos, memoria de ramas aislada
   y reutilización de rutas desde el estado real. El endpoint previsto de un plan
   no se presenta como un objetivo previo que haya causado la elección.
5. Ablaciones independientes: readout local frente a contextual; SIGReg sin
   hinge adicional; gate actual frente a 50 % conjunto y 50 % congelado.
   Retención de episodios, cachés, burn-in y backward son opciones del trainer,
   medidas frente a sus referencias, no componentes nuevos del agente.
6. Actor por imitación del MPC y después actor/crítico mediante imaginación
   corta. REINFORCE con ventajas detenidas, log-probabilidades diferenciables y
   mundo congelado. Cada acción propuesta condiciona la siguiente mediante su
   sucesor imaginado. Un generador independiente de subobjetivos y horizontes
   mayores quedan como experimentos, no bloquean la primera versión completa.

**Comparación:** tres semillas de nuevas etapas desde el checkpoint existente;
16.384 acciones reales por variante/semilla repartidas entre los diez pares
juego/nivel de `sc01-v1` y `si01-v1`. Controles intermedios pequeños a 4.096,
8.192 y 12.288 acciones; 32 episodios por nivel al cerrar desarrollo y 128 solo
para confirmar referencia y configuración seleccionada. Datos reales, etiquetas
docentes y trayectorias imaginadas mantienen procedencia separada. El benchmark
sintético no acredita generalización a juegos nuevos ni entrenamiento desde cero.

Se promueven mejoras de éxito, o variantes no inferiores (margen de dos puntos
porcentuales con intervalo emparejado del 95 %) que reduzcan coste: acciones,
latencia, parámetros, memoria o entrenamiento. Una simplificación elimina un
mecanismo o módulo completo, conserva los contratos y se documenta sin atribuirle
mejoras de velocidad no medidas. Resultados inconclusos conservan la referencia.
Las optimizaciones exactas requieren equivalencia numérica, no nuevos entrenos.

Los cambios del mundo invalidan sus cachés y requieren recalibrar las cabezas de
comportamiento. Se conservan dimensiones y pesos de origen; toda conversión es
explícita. CUDA usa `torch.compile`, el camino CPU permite verificar contratos y
se respeta el límite GPU de 100 W. Replay y checkpoints permanecen fuera de Git.
La arquitectura separará interacción, aprendizaje del mundo y del comportamiento,
con una definición por conjunto de pesos. La actualización visual del viewer se
integrará después de que el usuario confirme el fin del trabajo independiente
de la #54; no se monitoriza esa sesión.

El encoder, predictor recurrente y decoder visual aprenden de transiciones reales
del SDK. `model.agent.Agent` reúne los mismos módulos para observar, imaginar y
actuar; `Learner` puede actualizar su `Engine` sin crear otra implementación.
`evaluation.run --agent-checkpoint` y el adaptador Kaggle cargan ese agente con
pesos congelados. La política aleatoria sigue siendo el valor predeterminado de
los comandos que no reciben un checkpoint. Los checkpoints y el replay no se
suben a Git; el viewer conserva su contrato anterior durante esta iteración.

`model.proposals.ProposalPolicy` es opcional: produce propuestas categóricas y
coordenadas de clic, y contiene un crítico escalar de su política. Aprende por
imitación de decisiones MPC identificadas como tales; una variante posterior usa
REINFORCE, retornos lambda y dos pasos imaginados con mundo y evaluador congelados.
La segunda acción observa el sucesor imaginado de la primera. El MPC compara esas
rutas con sus alternativas y decide. El crítico del actor no sustituye a V/Q del
planner. No se ha incorporado un generador independiente de objetivos aprendido.
Antes de entrenar mediante imaginación, cada par juego/nivel debe tener muestras
reales a dos pasos, diagnósticos finitos y MSE latente no peor que mantener el
latente inicial (tolerancia numérica absoluta `1e-8`). Se reutiliza el diagnóstico
MC de la misma semilla sólo si coincide exactamente el fingerprint del mundo.
Este filtro provisional evita una degradación evidente; no acredita precisión
suficiente ni generalización. Si falla, la celda queda `deferred`, con sus semillas
y evidencia registradas, y se conserva la variante por imitación.

### Ejecución de las comparaciones

```sh
.venv/bin/python -u -m model.campaign \
  --checkpoint model/runs/active-training/latest.pt \
  --heads model/runs/goal-planner/multilevel/heads.pt \
  --output model/runs/architecture-evolution/full \
  > model/runs/architecture-evolution/campaign.log 2>&1
```

La campaña ejecuta secuencialmente referencias y variantes con tres semillas,
16.384 acciones por celda y diez pares sintéticos, sin leer reglas internas para
decidir acciones. Las variantes del mundo arrancan de los mismos pesos y comparten
recogida equilibrada/replay acotado; cada una recalibra las cabezas con experiencias
reales conservadas en RAM y recodificadas con sus pesos finales. El archivo de
calibración conserva todas las partidas recogidas, sin sobrescribir los éxitos
antiguos ni sobremuestrearlos; el replay del predictor sigue acotado por separado.
`--calibration-bytes` limita los bytes de observaciones originales (512 MiB por
defecto); si se supera, la ejecución falla explícitamente. Los latentes para V/Q
se crean después en CPU y consumen memoria adicional. El informe registra partidas,
acciones, progresos, fracasos, finales por presupuesto y targets censurados.
`rollout2_visual` añade `rollout_visual_weight=0.5` y se compara con `rollout2`:
decodifica el segundo latente predicho y supervisa su delta neto respecto al frame
real de inicio de ambos pasos. La memoria y el latente intermedios son predichos;
ninguna observación real intermedia entra en ese rollout. `local_decoder`
y `staged_freeze` se comparan con `rollout2`. `sigreg_only` se compara después con
`staged_freeze`: ambas entrenan encoder/predictor/decoder durante la primera
mitad de las acciones y congelan la representación en la segunda; sólo cambia
`variance_weight`, que pasa a cero. Así la comparación mide la regularización
cuando el encoder aprende, en vez de retirar una pérdida ya inactiva. El informe
y el progreso cuentan las actualizaciones conjuntas y con representación congelada.
Las variantes de
comportamiento conservan el mundo original; `double_q` y `click_mixed` se comparan
con `mc`. Los actores parten de las cabezas MC calibradas de su misma semilla.
Las propuestas de clic mixtas mantienen dieciséis candidatos y exploración uniforme.
En las variantes de comportamiento se reutiliza la evaluación del mundo de MC
sólo con la misma semilla y fingerprint exacto; el informe identifica su procedencia.
La evaluación de las decisiones de cada variante siempre se ejecuta de nuevo.
Las variantes que entrenan el mundo realizan su propia evaluación completa.
La selección final conserva una configuración completa que haya sido medida;
no combina automáticamente ganadores de comparaciones distintas sin evaluarlos juntos.

`report.json` distingue celdas terminadas, comparaciones inconclusas y confirmación
final. Volver a ejecutar el mismo comando omite celdas completas; una celda
interrumpida durante el aprendizaje vuelve a empezar desde su fuente, sin fingir
una restauración de replay. Al terminar el aprendizaje y la calibración, guarda
`pending-agent.pt` antes de evaluar. Si falla la evaluación, el mismo comando carga
ese agente y repite sólo la evaluación pendiente. No marca la celda como terminada
hasta completarla. Al incorporar este cambio de recogida hay que usar un directorio
nuevo: no se mezclan resultados de protocolos de calibración distintos.
Los enlaces duros `source-world.pt`/`source-heads.pt` conservan las fuentes sin
duplicar su almacenamiento. Sólo se retienen pesos representativos de la semilla
cero, candidata y seleccionada; no se elige la semilla con mejores resultados.
`--pilot` permite presupuestos menores para diagnosticar la ejecución, y nunca
promueve pesos ni constituye evidencia del protocolo completo.

Las comparaciones de éxito usan episodios emparejados por semilla/juego/nivel y
bootstrap estratificado; el intervalo describe estas nuevas etapas, no tres
preentrenamientos independientes. La elección del readout local conserva por ahora
los pesos de la torre contextual para comparación reversible: no se presenta como
reducción de parámetros. Su eliminación necesitaría una conversión explícita
posterior, justificada por los resultados. Burn-in es una aproximación optativa;
el backward ordinario/fragmentado y la caché congelada disponen de comprobaciones
numéricas. Una prueba piloto no demuestra generalización a otros juegos.
Las entradas compiladas de dinámica separan entrenamiento diferenciable e
inferencia; el decoder separa también readout local/contextual y gradientes activos
o desactivados. Comparten los mismos parámetros y conservan el límite normal
de caché. No se sustituye silenciosamente la compilación por ejecución eager.
El historial recurrente recalcula sus activaciones durante backward mediante
checkpointing, incluido el camino compilado. Así evita conservar una copia BF16
de las matrices recurrentes por cada paso de una partida larga, manteniendo la
cadena completa de gradientes; no introduce truncado de memoria ni cambia el
objetivo de aprendizaje.

El dashboard admite `--campaign-log model/runs/architecture-evolution/campaign.log`
además de sus logs existentes. Muestra el progreso por variante/semilla, presupuesto,
velocidad, errores a uno/dos/cuatro pasos, comparación identidad y éxito real,
separados de las pérdidas de actor/crítico. No hay inspecciones visuales automáticas.
La evaluación de comportamiento publica progreso al empezar y terminar cada partida
y aproximadamente cada dos segundos al completar acciones. La web muestra el bloque
juego/nivel, la partida actual, su presupuesto de acciones, partidas completadas,
victorias y tiempo registrado. Estos contadores no se suman al aprendizaje ni tratan
una partida en curso como derrota. Una decisión individual lenta puede retrasar el
siguiente aviso: la antigüedad del log sigue visible y no simula actividad.

La página `campaign.html`, enlazada desde el entrenamiento, compara toda la campaña
sin cambiar el selector de la ejecución individual. Añadir
`--campaign-report model/runs/architecture-evolution/full/report.json` al dashboard
permite recuperar los resultados completados incluso tras reiniciar la web o
descartar el tramo antiguo del log. Se lee el informe existente, sin crear otro
historial ni exponer trazas de episodios. La tabla muestra medias entre semillas
terminadas, avance de cada semilla, comparaciones controladas e intervalos del
95 %, candidata provisional y confirmación final separada. Las semillas pendientes
no cuentan como ceros y las evaluaciones intermedias no sustituyen resultados de
desarrollo completos. El informe se limita a 32 MiB y se relee sólo cuando cambia;
si no está disponible, la página lo indica sin reutilizar silenciosamente datos
anteriores. El servicio web puede reiniciarse sin reiniciar el entrenamiento.

El ensayo pequeño `python -m model.learning_probe --agent agente.pt --output
model/runs/architecture-evolution/full/learning-probe` compara pérdida latente a dos
pasos frente a añadir supervisión visual a dos pasos. Ambas ramas conservan la
supervisión visual original a un paso. Usa el `Learner` de producción,
los mismos pesos iniciales, datos, orden de muestreo y 120 actualizaciones por rama.
Encoder/decoder permanecen congelados, verificados al finalizar. Por defecto recoge
dos prefijos reales de hasta ocho acciones por juego/nivel para aprender y otros
dos con semillas disjuntas para desarrollo. Mide ambos conjuntos cada veinte
actualizaciones: mejorar sólo en los ejemplos de entrenamiento no acredita mejora
en partidas nuevas. No entrena V/Q ni el actor y no promueve pesos automáticamente.
CUDA conserva `torch.compile`; `--device cpu` permite un ensayo pequeño. Guarda los
pesos de cada rama antes de su evaluación final, sin exportar replay. La página de
comparación muestra sus curvas y cobertura aparte de la campaña y del diagnóstico
congelado; la página de entrenamiento enlaza la actividad en curso sin mezclarla con
los contadores de la ejecución seleccionada.

El ensayo siguiente usa `python -m model.streaming_probe --agent agente.pt --output
model/runs/architecture-evolution/full/streaming-probe`. Recoge 4.096 acciones
reales nuevas, repartidas por igual entre los diez pares sintéticos juego/nivel.
La primera receta incorpora partidas nuevas durante el aprendizaje; la segunda
recibe esos mismos prefijos en el mismo orden, desde RAM, sin acceso anticipado a
sus observaciones futuras. Se comprueban los pesos iniciales, el hash de los datos
y el número de actualizaciones de ambas ramas. Encoder y decoder siguen congelados.

Actualiza cada ocho acciones y al cerrar cada partida, con dos prefijos por lote
y ocho endpoints supervisados. Conserva la configuración y la tasa de aprendizaje
del checkpoint; sólo cambia la pérdida visual del segundo paso entre recetas.
La colección termina en el primer progreso, GAME_OVER o 128 acciones. Un corte por
cuota incompleta se censura; el progreso que entra en otro nivel se conserva como
etiqueta de valor, pero no como target visual del mundo.

Evalúa aproximadamente cada 1.024 acciones, al final de una partida, con dos
partidas distintas de hasta 32 acciones por par. El criterio de selección fijado
antes del ensayo es la media del error de delta visual de cada par y horizonte
(uno y dos pasos), incluyendo el checkpoint inicial. Guarda separadamente los
mejores y últimos pesos de cada rama; sólo la validación selecciona. Las gráficas
agregadas siguen ponderadas por la unión de píxeles. Se muestran además los casos
con cambios reales y los píxeles inventados por imagen cuando no hubo cambios.
La ausencia de cambios no acredita haber aprendido la dinámica.

Compara los pesos seleccionados con los iniciales en otras semillas de confirmación.
Después recalibra V/Q durante doce épocas con todas las experiencias nuevas,
reencodificadas con el mundo seleccionado y congelado, incluidos los éxitos tempranos.
Finalmente compara azar, Q directa y MPC con cuatro partidas por par, las mismas
semillas y 128 acciones por partida. Es un diagnóstico desde un único checkpoint,
no una demostración de generalización ni una promoción automática al agente activo.
El dashboard prefiere este ensayo cuando existe, muestra selección, confirmación,
cobertura y progreso dentro de cada partida. No mezcla sus curvas con el ensayo
anterior. Los artefactos quedan en directorios ignorados y no se guarda replay.

### Agente portable y contratos

La campaña puede pausarse entre ejecuciones completadas y conservar su `report.json`
para reanudar las pendientes. Para diagnosticar un predictor antes de ampliar el
entrenamiento, `python -m model.diagnostics --agents referencia.pt candidato.pt
--labels referencia candidato --output model/runs/architecture-evolution/full/diagnostics`
evalúa ambos agentes congelados; CUDA mantiene `torch.compile` y `--device cpu`
permite la comprobación pequeña. No actualiza pesos ni escribe replay.

El protocolo de diagnóstico usa los diez pares sintéticos juego/nivel. Por defecto
compara azar, Q directa y MPC con ocho episodios por par, las mismas semillas de
desarrollo nuevas y 128 acciones máximas. El coste temporal observado puede incluir
compilación: no se usa para promover una arquitectura. Otras dos sondas comparan
el delta visual predicho con conservar el tablero y con predecir otra acción.
Cada alternativa se ejecuta realmente desde la misma semilla y prefijo de acciones;
se comprueba la igualdad de todas las observaciones del prefijo y se repite el
sucesor para detectar falta de reproducibilidad. No se inspeccionan reglas ni estado
privado del juego. Los resultados de acciones equivalentes se separan de los pares
con sucesores visuales distintos. La sonda cubre dos semillas y cuatro estados
iniciales por par: es diagnóstico limitado, no confirmación ni generalización.

El informe se guarda atómicamente en el directorio indicado, que debe estar libre
de informes anteriores. Si está en `diagnostics/report.json` junto al informe de
campaña, la página de comparación muestra su progreso y tablas aparte. La web lee
como máximo 1 MiB y sólo publica resúmenes, sin pesos, rutas de checkpoints ni trazas.
Una nueva variante de entrenamiento se decide después de interpretar estas sondas;
el diagnóstico no selecciona ni promueve automáticamente ningún modelo.

```sh
.venv/bin/python -m model.agent_checkpoint \
  --world model/runs/active-training/latest.pt \
  --heads model/runs/goal-planner/multilevel/heads.pt \
  --output model/runs/agent.pt --mode value_mpc
```

La conversión comprueba la huella del mundo antes de unir pesos. La carga nueva es
estricta y guarda conjuntamente mundo, V/Q, configuración y actor/crítico opcional.
El estado de una partida/RNG sólo se incluye si se pide expresamente. En evaluación
y despliegue no se crea un optimizador ni se guarda replay. RESET consume una acción
confirmada y no repone el presupuesto global; las ramas imaginadas no lo consumen.

V/Q reciben presupuesto restante; la cabeza de eventos físicos no lo recibe.
Double-DQN selecciona candidatos con Q online y evalúa la misma acción con Q target
en `b-1`; la EMA se actualiza tras el optimizador. Progreso en la última acción vale
uno. Una recogida interrumpida se etiqueta como censurada, no como fracaso.
La Q de clic es la misma función Q condicionada por coordenadas, supervisada sólo
en acciones reales ejecutadas. La disponibilidad futura sigue desactivada por
defecto: haber actualizado su cabeza no acredita que esté calibrada.

`value_mpc` suma progreso esperado hasta el primer evento absorbente más V de hoja
cuando queda presupuesto; `leaf_value=q` usa la variante TD. Un endpoint decodificado
de la ruta seleccionada es una hipótesis visual del plan, no una recompensa ni una
intención previa inventada. Se conserva el modo experimental `goal_mpc` antiguo
para leer/comparar artefactos existentes, con su contrato de similitud diferenciado.

El diagrama refleja la variante en experimentación con `--pixel-latent`,
`--broadcast-conditioning`, `--local-dynamics`, y escalas residual/posicional .1.
Sus operaciones enlazan los 491 tensores reales compartidos entre llamadas. La
configuración del checkpoint identifica qué variante ejecuta cada entrenamiento;
el diagrama no demuestra convergencia ni generalización entre juegos.

## Referencia anterior: objetivo y planner con el mundo congelado

Esta sección describe el experimento `behavior_experiment` y sus artefactos
existentes. La campaña nueva añade los contratos de presupuesto y propuestas
descritos arriba; no atribuye retrospectivamente esos contratos a pesos antiguos.

`planning.py` añade 603.013 parámetros nuevos (`GoalValue`), frente a los
215.984.848 del mundo existente. El encoder visual, su memoria de animación,
el predictor recurrente y el decoder se cargan del checkpoint, pasan a `eval()`
y `requires_grad=False` y no tienen optimizador. Se compara un SHA-256 de todos
sus tensores antes/después del experimento. No se modifica el checkpoint fuente.

El flujo de esta primera fase es:

1. Observar el juego y obtener el estado real `(z, h)` con el encoder congelado.
2. Recoger acciones y resultados reales. El SDK indica avance de nivel, `WIN`
   y `GAME_OVER`; no se inspeccionan reglas, colores o posiciones privadas.
3. Entrenar solamente los nuevos readouts: probabilidad de progreso/final y
   valores `V(s)` y `Q(s,a)` de éxito descontado. Las etiquetas son eventos y
   retornos de partidas reales; la metadata futura nunca entra como feature.
4. Comparar la elección directa por `Q` con MPC: predecir candidatos mediante
   `z_next = z + delta`, actualizar su memoria independiente y valorar futuros.
5. Ejecutar una acción y reconstruir el estado con la observación real recibida.
   Las ramas imaginadas no se guardan como experiencia real.

El objetivo actual es **predecir el éxito señalado por el SDK**. No hemos
implementado descubrimiento de reglas simbólicas ni disponemos de una imagen
objetivo. `V/Q` aprenden retornos Monte Carlo de horizonte finito de la conducta
recogida, no valores óptimos demostrados. La terminación por presupuesto usa
retorno cero más allá del límite, pero no se etiqueta como `GAME_OVER`.

El readout usa LayerNorm sobre tokens, convoluciones espaciales de 64 canales,
pooling 4×4 **solo en el nuevo evaluador**, una proyección de memoria a 64 y un
MLP 1088→256→128 con SiLU. No cambia la resolución ni los pesos del encoder.
Una cabeza sigmoide produce `V`; otra recibe además las 12 features de acción
y produce tres logits de evento y un logit `Q`. Se optimizan CE de eventos sin
ponderación y BCE contra retornos descontados (`gamma=.95`).

MPC combina recompensa de progreso prevista, probabilidad de continuar y valor
al final del horizonte. Conserva cuatro ramas, compara horizontes de uno/dos
pasos y replantea tras cada acción. Considera todos los IDs legales ordinarios
y ocho coordenadas uniformes cuando existe clic. Reutiliza la legalidad actual
dentro de la búsqueda: **no conoce la legalidad futura**. El decoder no participa.
`Q` es una comparación barata que prescinde de simular futuros.

Ejecutar desde la raíz, después de detener cualquier entrenamiento que vaya a
modificar el checkpoint usado como referencia:

```sh
.venv/bin/python -u -m model.behavior_experiment \
  --checkpoint model/runs/active-training/latest.pt \
  --output model/runs/goal-planner/new-run \
  --game sc01-v1 --level 2 --device cuda \
  --train-episodes 128 --rounds 2 --epochs 12 \
  --dev-episodes 32 --final-episodes 128
```

CUDA exige `torch.compile` y usa BF16 cuando está disponible; `--device cpu`
ejercita el mismo contrato sin compilación. Las primeras compilaciones entran
en los tiempos medidos. Cada ronda conserva el mejor candidato aprendido de desarrollo,
recoge experiencia real nueva con exploración epsilon=.25 y vuelve a ajustar
los readouts. Sin inicialización, la primera ronda recoge al azar. Desarrollo y evaluación final
usan semillas distintas del entrenamiento. La evaluación final se hace una
vez tras seleccionar el candidato; no entrena ni guarda replay.

`report.json` contiene recuentos, semillas, acciones, tiempos e intervalos de
Wilson; `heads.pt` solo añade pesos de comportamiento, la huella del mundo y
configuración. Al completar cada ronda guarda el mejor candidato de desarrollo;
al terminar añade su evaluación final. El replay de latentes reales es FIFO en RAM, máximo 8192
transiciones por defecto (aproximadamente 4 GiB para este modelo), y desaparece
al salir. La memoria recurrente `h` se reinicia por partida. Se imprimen eventos
JSON de recogida, ajuste y evaluación. No escribe al dashboard del trainer del
mundo ni arranca automáticamente el currículo anterior. El dashboard puede leer
su log independiente mediante `--behavior-log`, como se describe abajo.

Para evaluar esos readouts sin entrenamiento adicional:

```sh
.venv/bin/python -u -m model.behavior_experiment \
  --checkpoint model/runs/active-training/latest.pt \
  --heads model/runs/goal-planner/new-run/heads.pt \
  --output model/runs/goal-planner/evaluation \
  --game sc01-v1 --level 2 --device cuda --seed-offset 20000
```

Para continuar aprendiendo desde sus pesos, usar **`--init-heads`**, incompatible
con `--heads` (que sigue siendo solo evaluación):

```sh
.venv/bin/python -u -m model.behavior_experiment \
  --checkpoint model/runs/active-training/latest.pt \
  --init-heads model/runs/goal-planner/more-experience/heads.pt \
  --output model/runs/goal-planner/continuation \
  --game sc01-v1 --level 2 --device cuda --seed-offset 30000 \
  --train-episodes 512 --rounds 4 --epochs 12 --max-records 16384 \
  --dev-episodes 64 --final-episodes 256 --horizons 1 2
```

Con `--level` se exige la misma huella del mundo, juego y nivel. El optimizador y la memoria de
experiencia empiezan vacíos: es una continuación desde pesos, no la restauración
exacta de una ejecución interrumpida. La política inicial conserva su configuración
y participa como candidata (iteración 0); las nuevas usan el beam solicitado y el
descuento/muestreo de clics inicial. El entrenamiento usa ese mismo descuento.
Desde la primera ronda se recogen acciones con la mejor política y exploración.
Si los candidatos nuevos empeoran en desarrollo, conserva los pesos iniciales.

La comparación final usa las mismas semillas nuevas para azar, política inicial
y política elegida, sin actualizaciones. La fila/serie **Inicial** del panel es
una referencia separada; no sustituye el resultado del candidato nuevo.
`initial_head_fingerprint` y `head_fingerprint` identifican los pesos comparados.
El mundo, encoder y decoder siguen completamente congelados.

Para entrenar **unas mismas cabezas en varios niveles**, `--levels` sustituye a
`--level` y permite transferir las cabezas entre niveles del mismo juego:

```sh
.venv/bin/python -u -m model.behavior_experiment \
  --checkpoint model/runs/active-training/latest.pt \
  --init-heads model/runs/goal-planner/continuation/heads.pt \
  --output model/runs/goal-planner/multilevel \
  --game sc01-v1 --levels 3 4 5 --device cuda --seed-offset 40000 \
  --train-episodes 128 --rounds 2 --epochs 12 --max-records 16384 \
  --dev-episodes 32 --final-episodes 128 --horizons 1 2
```

Los presupuestos de partidas se aplican **por nivel**: este comando recoge 768
partidas entre las dos rondas, evalúa 96 partidas por candidato en desarrollo y
384 por política al final. Alterna los niveles por partida y reinicia la memoria
recurrente en cada una. El ID del nivel se usa para el entorno y las métricas;
no entra en la red. Solo hay una política y un checkpoint compartidos.

El máximo de registros se reparte en cuotas FIFO por nivel (5462/5461/5461 con
este comando), que después se mezclan para el ajuste. Así ningún nivel expulsa
las experiencias de los demás. Sigue siendo memoria temporal, sin grabaciones
de replay en disco. El mundo debe coincidir con la huella de las cabezas iniciales
y permanece congelado; se comprueba su huella y la ausencia de gradientes después
de cada ajuste y al terminar.

Desarrollo y evaluación final presentan resultados `per_level` y un agregado
con el mismo número de partidas por nivel. La selección usa las victorias del
agregado de desarrollo, con coste de acciones/decisión para desempatar, conservando
la política inicial como candidata. Un agregado mejor puede esconder un retroceso
en un nivel: la tabla de **Evaluaciones** del panel muestra ambos. Cada política se
evalúa sobre los mismos pares nivel/semilla; la evaluación final nunca decide
qué checkpoint guardar. Los pesos iniciales no se sobrescriben. No cambia la
arquitectura de inferencia ni `architecture.json`.

Se rechaza una huella de mundo distinta y un output que ya contenga resultados.
Para nuevas iteraciones, elegir semillas nuevas con `--seed-offset`; no tratar
como evaluación final un conjunto usado para decidir cambios. El experimento
admite por ahora **solo los dos juegos sintéticos del repositorio**. Las clases
del planner consumen la interfaz genérica del SDK; la CLI falla explícitamente
para otros juegos hasta conectar su admisión a los splits y niveles reservados.

### Primera prueba con pesos existentes

Checkpoint del mundo: acción **143840**, huella de tensores
`72b2f1dbbf74bc0c4faf02ae16a7dcb50fb04696523bfa7f3c04d3f1138248ad`.
Square Cross, nivel sintético aislado 2, máximo 128 acciones por partida,
RTX 4070 Ti compartida, PyTorch 2.6.0, BF16 y `torch.compile`.
Las tres pasadas de ajuste modificaron exclusivamente los nuevos readouts.

| Experimento | Entrenamiento real | Candidato elegido en desarrollo | Azar, final | Aprendido, final | Acciones totales azar / aprendido |
| --- | --- | --- | --- | --- | --- |
| Inicial | 2 rondas × 128 partidas, 12 épocas por ronda | Q, primera ronda | 28/128 (21,88%) | 31/128 (24,22%) | 3460 / 3165 |
| Más experiencia | 512 partidas, 13046 transiciones, 20 épocas | MPC, horizonte 1 | 32/128 (25%) | 36/128 (28,13%) | 3211 / 2921 |

La segunda prueba usó `--seed-offset 10000 --max-records 16384`, con 64 partidas
de desarrollo. En ese desarrollo, azar ganó 17/64; Q, MPC-1 y MPC-2 ganaron
11, 14 y 12. La selección es entre candidatos aprendidos, no un reemplazo
automático de la política aleatoria. En evaluación final, los pares discordantes
fueron 24 victorias exclusivas del azar frente a 28 exclusivas del planner.
**No demuestra una ventaja estadística ni que el objetivo esté aprendido.**

El MPC seleccionado tardó 30,41 s para 2921 acciones, unos 6,02 ms por decisión
y 96 acciones/s incluyendo la actualización del estado real. El azar no ejecuta
las redes, por lo que su velocidad sirve solo como referencia del entorno.
Los errores visuales del predictor no son estas tasas de victoria. Ambos
experimentos verificaron la huella del mundo sin cambios; los pesos nuevos y
reportes detallados quedan solo en `model/runs/goal-planner/`. No se deja un
entrenamiento indefinido en marcha ni se reanuda el currículo del mundo.

### Continuación desde las cabezas guardadas

Con el mismo mundo congelado, se continuó desde `more-experience/heads.pt`:
cuatro rondas × 512 partidas, 12 épocas por ronda, FIFO de 16384 transiciones,
64 partidas de desarrollo y 256 finales, `--seed-offset 30000`. Se recogieron
**50485 acciones reales** (2048 partidas, 450 avances de nivel). El optimizador
se inicializó de nuevo y no se recuperó replay anterior.

| Desarrollo (victorias / 64) | Q | MPC-1 | MPC-2 |
| --- | --- | --- | --- |
| Ronda 1 | 10 | 17 | 12 |
| Ronda 2 | 6 | 9 | 12 |
| Ronda 3 | **26** | 17 | 19 |
| Ronda 4 | 18 | 20 | 21 |

En ese desarrollo, la inicial MPC-1 ganó 13 y azar 19. Se seleccionó Q de la
tercera ronda antes de abrir el conjunto final. La cuarta no sobrescribió esos
pesos. Las 256 semillas finales fueron las mismas para las tres políticas:

| Política | Victorias | Acciones totales | Acciones por victoria, incluidos fallos | ms/decisión |
| --- | --- | --- | --- | --- |
| Azar | 63/256 (24,61%) | 6967 | 110,59 | 0,002 |
| MPC-1 inicial | 53/256 (20,70%) | 6572 | 124,00 | 6,08 |
| Q seleccionado | 73/256 (28,52%) | 6223 | 85,25 | 1,09 |

Q ganó 47 pares que perdió la inicial y perdió 27 que esta ganó. Frente al azar,
los pares exclusivos fueron 50 a favor de Q y 40 del azar: **todavía no demuestra
una ventaja clara sobre el azar**. La selección directa fue mejor en desarrollo
que la imaginación de uno/dos pasos; esto no demuestra que aumentar el horizonte
vaya a resolver la diferencia. No hay evaluación de generalización a otros juegos.

La evaluación final se hizo con la RTX 4070 Ti limitada a **150 W**: Q tardó
33,08 s (188,14 acciones/s) e inicial MPC 68,84 s (95,47 acciones/s). Son velocidades
de evaluación, no de backpropagation. El límite se redujo desde 250 W durante la
recogida de la cuarta ronda; los costes entre rondas no son una comparación
controlada de velocidad. Posteriormente se bajó a 100 W por petición del usuario;
estas cifras no son un benchmark a 100 W.

La huella del mundo anterior se verificó idéntica y ningún parámetro del mundo
recibió gradiente. La cabeza pasó de
`e88ba13d86eb7b3321209c6fa63546d05ad91f71207d2b8be3649394642d69e4` a
`9bacb378189a700fea416b173577909b3f0902aad77bc0cb766d9a753a921b3f`.
Los resultados y pesos están en el directorio local ignorado
`model/runs/goal-planner/continuation/`. No cambia la arquitectura de inferencia
de `architecture.json`; cambia la inicialización y selección del experimento.
La partida de inspección independiente con seed **9357821** terminó en WIN tras
12 acciones. Se exportó con predicción visual y diagnóstico Q anteriores a cada
acción; Q registra una elección, no una trayectoria imaginada de varios pasos.
La muestra no se usó para seleccionar pesos ni para calcular las tasas anteriores.

### Referencias para las siguientes iteraciones

Lectura del 14-10-2026; son referencias e hipótesis, no capacidades implementadas.

- [V-JEPA 2 (junio 2025)](https://arxiv.org/abs/2506.09985): MPC con CEM y
  distancia a una imagen objetivo en el espacio latente. Ejecuta la primera
  acción y vuelve a observar. Nos sirve el bucle de control; el paper presupone
  subobjetivos visuales, no descubre el objetivo desconocido de un juego.
- [Dreamer 4 (septiembre 2025)](https://arxiv.org/abs/2509.24527): durante RL
  imaginado actualiza cabezas de política/valor con el transformer congelado.
  La fase anterior sí adapta el modelo al incorporar política y recompensa;
  no equivale a añadir cualquier actor a nuestros pesos sin esa preparación.
- [Value-guided action planning with JEPA world models
  (preprint diciembre 2025, arXiv enero 2026)](https://arxiv.org/abs/2601.00844):
  entrena la geometría latente con valor condicionado a objetivos y prueba
  cuasidistancias. Sus experimentos señalan que predecir latentes no garantiza
  buena planificación. Modifica el encoder; una cabeza de valor externa sobre
  nuestro encoder congelado sería una adaptación nuestra, no una reproducción.
- [LeWorldModel (marzo 2026)](https://arxiv.org/abs/2603.19312): JEPA compacto
  con aprendizaje estable y planificación latente. Es una referencia de coste
  y representación; adoptar su entrenamiento implicaría cambiar el mundo,
  fuera del alcance de esta fase.
- [Hallucination in World Models is Predictable and Preventable
  (junio 2026)](https://arxiv.org/abs/2606.27326): estudia cobertura de estados/
  acciones y señales de fallo en modelos generativos. Motiva comprobar los
  planes en el entorno real; no demuestra que nuestro fallo tenga esa causa.
- [V-JEPA Policy (29 septiembre 2026)](https://arxiv.org/abs/2609.37250):
  encoder V-JEPA 2.1 congelado, pero predictor nuevo y experto de acciones
  entrenados conjuntamente con demostraciones e instrucciones. No mantiene
  congelado todo el predictor como pedimos aquí.
- [JEPA-TTT (30 septiembre 2026)](https://arxiv.org/abs/2610.00722): adapta el
  predictor ante cambios de dinámica, manteniendo encoder y recompensa fijos.
  Es relevante para adaptación futura, pero contradice la congelación actual
  del predictor y no se activa en este experimento.

## Instalación y ejecución

Desde la raíz del repositorio, con Python 3.12:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python -r model/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m model train
```

Dependencias opcionales para la RTX 4070 Ti (Ada):

```sh
uv pip install --python .venv/bin/python -r model/requirements-cuda.txt
.venv/bin/python -m model train --device cuda --compile --no-fp8 --optimizer-bits 8
```

La comparación experimental usa BF16 y momentos AdamW de 8 bits. `--fp8` activa
otra variante que debe evaluarse por velocidad **y** calidad; no es un requisito
para usar Tensor Cores ni una mejora garantizada. FP8 usa escalado dinámico por
tensor de torchao 0.8.0, compatible con PyTorch
2.6: convierte las 156 proyecciones Q/K/V/salida y MLP de los bloques Transformer.
Usa E4M3 para entradas/pesos y E5M2 para operandos de gradiente; conserva pesos
maestros FP32. Las proyecciones de entrada/salida del modelo, las recurrencias y
las normalizaciones quedan fuera de la conversión. La suma `z + delta`, las
pérdidas y las estadísticas anticolapso conservan FP32; el resto usa autocast BF16.
Se rellena internamente la dimensión de reducción cuando FP8 exige múltiplos de
16, sin modificar el tablero ni sus objetivos. FP8 requiere CUDA >= SM 8.9 y
`torch.compile`; no se emula ni se cambia silenciosamente a eager.

AdamW de bitsandbytes 0.45.5 conserva momentos cuantizados por bloques en tensores
grandes y FP32 en tensores pequeños; no cuantiza los pesos maestros. CPU usa AdamW
de 32 bits. `--no-fp8 --optimizer-bits 32` permite una comparación CUDA compilada.
Estos formatos no garantizan mayor velocidad ni igual convergencia: se observan
por separado el tiempo por acción, el uso de memoria y el error de delta.

Se elige CUDA cuando está disponible y se activa `torch.compile`, SDPA, BF16
cuando el dispositivo lo soporta y checkpointing de activaciones. AdamW de 32
bits usa la implementación fusionada; el de 8 bits usa kernels de bitsandbytes.
La primera actualización incluye el coste de compilación. En el entrenamiento
CUDA se mantiene siempre `torch.compile`; `--no-compile` se reserva al desarrollo
en CPU. Una compilación fallida no se oculta como un cambio automático de modo.
El rendimiento depende del hardware y de las longitudes de las partidas.
No se afirma que compile siempre sea más rápido; evita recompilar usando la misma
configuración. La medición de rendimiento empieza al completar la primera acción.
Los microbatches del encoder usan almacenamiento independiente para evitar que
el compilador los especialice según el tamaño del prefijo del que proceden.
Durante el entrenamiento, la memoria inicial cero tiene habilitado el cálculo
de gradiente igual que los estados recurrentes posteriores; sigue siendo una
constante efímera, fuera del optimizador. Esto elimina otra causa de
recompilación sin alterar la actualización matemática del modelo.

El modelo **base sin las opciones espaciales** tiene 214.143.104 parámetros:

| Componente | Configuración | Parámetros |
|---|---|---:|
| Visual encoder | 12 bloques, ancho 768, 12 cabezas; memoria de animación 1024 por patch | 91.525.376 |
| Dynamics | Memoria global 2048; predictor de 6 bloques, ancho 768; latente P×1024 | 64.929.152 |
| Visual decoder | 8 bloques, ancho 768, 12 cabezas; 16 colores | 57.688.576 |

`--residual-scale` multiplica las ramas de atención y MLP antes de cada suma
residual Transformer; `--position-scale` multiplica los códigos sinusoidales.
Son factores fijos, guardados en la configuración del checkpoint, que permiten
moderar la mezcla global inicial sin quitar capas ni píxeles. No afectan al
coeficiente de la suma predictiva: siempre se mantiene `z_hat = z + d_hat`.
Los checkpoints restauran sus propios factores, sin permitir cambiarlos
silenciosamente al reanudar o validar.

`--pixel-latent` conserva 64 características aprendidas por píxel nativo. Un
`Linear(35,64)` compartido codifica color, delta observado de animación y validez;
los 16 píxeles de cada patch forman sus 1024 características latentes. Se suma
el contexto recurrente de animación multiplicado por `residual_scale` y se
aplica LayerNorm por patch. No hay pooling ni un atajo de píxeles al decoder.
El decoder suma al latente una corrección contextual `Linear(768,1024)` escalada
y utiliza un clasificador compartido `Linear(64,16)` por píxel. Esta variante
tiene **214.737.040 parámetros**: encoder 91.527.680, dynamics 64.929.152 y
decoder 58.280.208.

`--broadcast-conditioning` añade las proyecciones de acción y memoria a todos
los tokens espaciales del predictor, además de conservar sus dos tokens globales.
`--local-dynamics` requiere `--pixel-latent` y añade una rama aprendida de
convoluciones sobre la rejilla latente nativa: 64→128 canales con kernel 3×3,
condicionamiento lineal de memoria/acción, tres bloques residuales
GroupNorm(8)→SiLU→Conv3×3→GroupNorm(8)→SiLU→Conv3×3, y salida 1×1 de 128→64
inicializada a cero. Añade posiciones 2D explícitas a la rama Transformer y
combina `d_hat = local + residual_scale * global`. El estado sigue avanzando
como `z_hat = z + d_hat`, con coeficiente uno. El total es **215.984.848 parámetros**.
La atención global y la memoria permanecen activas para dependencias no locales;
las convoluciones no codifican reglas, objetos, colores o movimientos de un juego.
Mejorar Square Cross no establece generalización a otros juegos.

`--profile small --device cpu --no-compile` usa la misma computación con
74.872 parámetros para desarrollo. No es un sustituto de capacidad equivalente.
`--frame-chunk` limita frames por llamada espacial (4 por defecto). Cada
actualización reconstruye prefijos causales independientes. `--batch-episodes`
controla cuántas partidas reúne y `--loss-steps` cuántas transiciones finales
supervisa en cada prefijo. El perfil base no se reduce automáticamente al encontrar
límites de memoria.
La compilación puede aumentar el pico de memoria. Si no cabe en una GPU compartida,
hay que ajustar el uso de memoria del cálculo compilado, conservando BF16, SDPA
y el tamaño de la red; desactivar la compilación no es la solución de entrenamiento.

## Flujo y entrenamiento

1. El adaptador abre un episodio con semilla nueva y obtiene la observación SDK.
   No añade una acción artificial a la primera observación.
   Square Cross permuta toda su paleta, incluido el fondo, según la semilla;
   mantiene los colores durante la partida. También varía la posición inicial
   del conjunto en los niveles 1–2, conservando sus reglas. Son propiedades del entorno:
   el modelo no recibe el mapa de colores ni etiquetas de objetos.
2. El comportamiento aleatorio elige una acción de `available_actions`; ACTION6
   incluye coordenadas SDK uniformes en 0…63. Prepara el resultado real una sola
   vez en el dispositivo del modelo y lo añade al buffer de la partida actual.
3. Actualiza cada `--train-every` acciones, además de la primera y GAME_OVER/WIN.
   Un avance nativo de nivel también vacía cualquier actualización pendiente del
   prefijo válido, antes de liberar el replay del nivel anterior.
   El modo por acción sigue disponible con `--train-every 1`. El lote incluye el
   prefijo actual y, si se configura replay, prefijos de otras partidas reales.
   Vuelve a codificar cada **prefijo completo desde su propio inicio** con los pesos
   actuales. Conserva todos los frames de animación.
   Los deltas entre frames de una animación forman parte de las entradas del
   encoder, junto con los colores y una máscara de píxeles válidos. El primer
   frame de cada respuesta usa un símbolo absoluto.
4. Conserva **un latente por patch**, `z[P,1024]`, con
   `P=ceil(H/4)*ceil(W/4)`. Un patch concatena los valores de sus 16 píxeles;
   no los promedia. No hay pooling espacial del estado. La memoria de animación
   avanza por posición, y el latente final tiene LayerNorm por patch.
   Reconstruye la memoria causal global `h` desde el inicio de cada partida. El predictor
   recibe `z_t, h_t, a_t` y calcula `h_(t+1), d_hat_(t+1)`;
   **`z_hat_(t+1) = z_t + d_hat_(t+1)`**, sin normalización posterior a la suma.
   Para actualizar `h` se resume `z`, pero los P latentes completos llegan al
   predictor. Sus tokens son los P patches más uno de memoria y uno de acción.
5. Entrena encoder y dynamics con `(1-w) * MSE(d_hat, z_(t+1)-z_t) + w * SIGReg
   + variance_weight * variance_floor`, además de los gradientes visuales del siguiente paso.
   `--sigreg-weight` controla `w` (por defecto .002); los pesos activos se imprimen al iniciar.
   Ambos latentes reales conservan gradiente. SIGReg usa al menos 2
   observaciones reales de igual geometría, proyecciones aleatorias y cálculo FP32.
   `--min-observations` controla solamente este mínimo de regularización: nunca
   bloquea una actualización predictiva/visual válida.
   No hay profesor EMA ni etiquetas tomadas de trayectorias imaginadas. SIGReg
   integra ambos lados de la función par. Muestrea hasta 16 posiciones, siempre
   las mismas dentro de un batch de geometría común, y calcula estadísticas
   **entre observaciones**: distintos códigos de posición no demuestran diversidad.
   La penalización de varianza es `mean(relu(1-sqrt(var(z,obs)+1e-8)))`, con
   varianza poblacional por posición/característica. `--variance-weight` controla
   su peso; no normaliza artificialmente las métricas de validación.
6. El decoder recibe los P latentes predichos y produce 16 distribuciones de
   color por patch, una por píxel. La probabilidad del color anterior se asigna
   a `KEEP=16`; los otros colores indican reemplazos. Compara este **delta
   predicho con el delta real** mediante NLL, equilibrando celdas cambiadas y
   no cambiadas. Cada grupo aporta la mitad de la pérdida cuando ambos están
   presentes; dentro de cada grupo todos los píxeles pesan igual, sin ponderar
   por frecuencia de color. Si solo hay un grupo, usa su media ordinaria.
   `--false-change-weight` (1 por defecto) multiplica exclusivamente la NLL de
   las celdas estáticas de la **rama de predicción**. Por ejemplo, 2 duplica esa
   contribución sin reducir la de los cambios reales; no modifica las ramas de
   reconstrucción, el término de unión ni los umbrales de validación. Es una
   opción experimental: reducir cambios inventados no demuestra una mejora si
   a cambio se omiten más cambios reales. La validación informa por separado
   `extra_changes`, `missed_changes`, `wrong_replacement_colors` y `correct_changes`.
   Esa asignación es biyectiva: la NLL equivale a CE de color;
   cambiar las etiquetas por sí solo no evita el colapso.
   Lo anterior describe el modo histórico `prediction_delta_loss: "balanced"`.
   El modo anterior `"union"`, aún disponible, se aplica a las ramas predictivas
   de uno y tres pasos. Con `x` anterior, `y` real y `yhat=argmax(logits)`:
   `U = (y != x) OR (yhat != x)` y
   `L_union = sum(where(U, -log softmax(logits)[y], 0)) / max(1, |U|)`.
   Se normaliza por imagen y después se promedian las imágenes. La máscara es
   discreta, sin gradiente a través de argmax; la NLL sí propaga gradiente.
   Una unión vacía aporta cero diferenciable. No usa top-K, frecuencia de color
   ni `false_change_weight`: todos los píxeles de la unión pesan igual.
   Este modo no cambia la NLL equilibrada de las ramas de reconstrucción.
   La receta incremental actual selecciona `"recall_focal_capped"`: suma el doble de la NLL de cambios
   reales y la NLL estática ponderada por `false_change_weight*(1-q)^2`, dividida
   por `max(32, número de cambios reales)`, con `q=softmax(logits)[y]` y T=1.
   El modo histórico `recall_focal` conserva el divisor `max(1, cambios reales)`.
   El suelo 32 atenúa por igual todos los términos de una imagen con pocos cambios;
   no altera su preferencia relativa entre omisiones, colores erróneos e invenciones.
   El loss latente mantiene su escala, por lo que aumenta su peso relativo en esas imágenes.
   No selecciona píxeles mediante argmax ni diferencia el divisor; sí diferencia
   los pesos suaves. Su contribución estática disminuye de forma continua al
   acertar con más confianza. No es estrictamente invariante a añadir fondo con
   incertidumbre, y no altera la métrica de evaluación basada en colores argmax.
   El modo `focal` anterior conserva peso 1 para cambios reales; `recall_focal`
   usa peso 2. Esta preferencia por recuperar cambios es experimental: no identifica
   objetos ni garantiza preservar piezas. A confianza 0,9/0,1 y 56 cambios reales,
   omitir 25 cambios aporta más pérdida que acertarlos e inventar 30 estáticos;
   acertar todo sin inventar sigue siendo mejor. Los errores de color en cambios
   reales reciben también peso 2. El MSE latente mantiene su peso anterior.
   El aprendiz conjunto histórico y la reconstrucción añaden un término
   diferenciable de error sobre la unión de deltas. El predictor de la campaña
   incremental (`OnlinePredictor`) no lo utiliza, ni a uno ni a tres pasos.
   Si `c` es el número de celdas realmente cambiadas, `f` la suma de probabilidades de
   cambio falso en celdas estáticas y `m` la suma de probabilidades de error en
   celdas cambiadas, usa `(m+f)/(c+f)`. Así los cambios inventados pesan según
   el tamaño del delta. En frames sin cambios reales usa la probabilidad media
   de cambio falso para conservar gradiente. `--delta-overlap-weight` controla
   este término (1 por defecto, 0 lo desactiva); la métrica de validación sigue
   usando argmax y su unión real, sin suavizarla.
7. El mismo decoder reconstruye también el sucesor desde su latente **real**,
   con peso 1, igual que la rama predicha. Ambas pérdidas propagan gradiente
   al encoder; la predicha también al predictor. Los targets son siempre
   píxeles reales. La reconstrucción permite comprobar que el latente conserva
   información visual mientras el predictor aprende. No es una predicción causal.
8. Una lectura auxiliar **local por patch** reconstruye los píxeles del sucesor
   real usando la proyección latente, LayerNorm y clasificador del decoder,
   compartidos. Omite posiciones y atención entre patches. Obliga a que cada
   latente contenga información de sus propios píxeles: cambiar otro patch no
   puede alterar esta lectura. `--local-reconstruction-weight` controla su peso
   (1 por defecto; 0 la desactiva). No añade parámetros ni forma parte de la
   predicción en inferencia; sus targets y gradientes son exclusivamente reales.
   Con `--pixel-latent`, esta rama lee directamente los 64 canales de cada píxel
   usando el clasificador compartido, sin proyección contextual ni normalización
   adicional del decoder.

El objetivo total suma los términos latentes y las tres pérdidas visuales
(NLL más error de unión suave por rama). AdamW usa `lr=3e-5` por
defecto. Los dos optimizadores conservan conjuntos de parámetros disjuntos,
pero todas las pérdidas se calculan con los mismos pesos previos a la actualización.
Para limitar memoria, cada microbatch del decoder acumula gradientes en una hoja
temporal; al terminar se propagan explícitamente por el grafo original de ambos
latentes. El encoder conserva las salidas del prefijo y recalcula sus
activaciones internas durante backward, para no retener todos los intermedios
de la partida a la vez. Ese puente **no corta el gradiente**.
Se comprueban y recortan ambos
conjuntos de gradientes antes de actualizar ningún peso.

Los píxeles del tablero anterior no entran a la red del decoder: solo determinan
la geometría y la interpretación de sus colores como deltas. La suma residual
opera por patch y sigue permitiendo imaginar sin generar píxeles.

Los estados recurrentes de validación avanzan con observaciones reales. El
entrenamiento aleatorio no calcula inferencias en vivo que no va a consumir: las
reconstruye en los batches. `Engine.rebuild` permite reconstruir un estado en vivo
después de actualizar los pesos. No existe caché persistente de latentes obsoletos.
La recurrencia de animación agrupa observaciones de igual geometría y número de
frames para procesarlas en paralelo. Solo avanza por el eje temporal de cada
animación: el eje de batch y los patches mantienen memorias independientes.
Esta operación también se compila en CUDA.

El buffer mantiene hasta `--replay-episodes` partidas reales. Sus observaciones SDK
originales siguen en CPU y los píxeles, features visuales y acciones preparados
residen en GPU durante el entrenamiento CUDA. Solo reutiliza entradas inmutables:
no hay caché de latentes ni grafos. `--replay-bytes` limita conjuntamente los
tensores preparados (128 MiB por defecto); antes de excederlo expulsa partidas
antiguas. Una partida que no cabe por sí sola produce un error explícito. CPU usa
la misma ruta con tensores CPU. La transición terminal se entrena antes del reset.
Cada nueva partida empieza con memoria recurrente vacía; las partidas almacenadas
se muestrean por separado, sin concatenar sus historias. Los prefijos antiguos se
truncan al azar y el actual llega hasta su última acción. Los pesos se conservan,
pero el replay se vacía al cambiar de nivel o reiniciar el proceso. No se persiste.
Con `--replay-episodes 1 --batch-episodes 1 --loss-steps 1` se conserva el modo
limitado a la última transición de la partida actual. Las semillas y acciones del
entorno tienen un generador independiente del muestreo de replay.
Una respuesta sin frames se conserva, avanza la memoria con la acción cuando
hay estado y no inventa un objetivo visual. El encoder admite geometrías nativas
rectangulares y padding interno; predecir cambios de tamaño de canvas queda fuera
de este primer decoder. Tales transiciones se omiten del loss y hacen fallar el
criterio de validación, en lugar de contarlas como aciertos.

## Diagnósticos de ajuste

El diagnóstico aislado del encoder/decoder usa pares reales del SDK y un conjunto
pequeño fijo, sin predictor ni regularización latente. Permite distinguir un
problema de reconstrucción de uno de dinámica:

```sh
.venv/bin/python -m model.experiment --steps 160 --pairs 8 \
  --learning-rate .0003 --residual-scale .1 --position-scale .1 --pixel-latent
```

Para aislar la dinámica después de obtener una buena reconstrucción, se pueden
congelar encoder y decoder de un checkpoint y ajustar solo el predictor sobre
primeras transiciones reales. Los latentes de este diagnóstico permanecen fijos:

```sh
.venv/bin/python -m model.experiment --task dynamics --checkpoint model/runs/entrenamiento/latest.pt \
  --steps 160 --pairs 8 --learning-rate .0003
```

El decoder congelado sigue propagando gradientes al predictor. Se informa el
desglose visual y el MSE latente, comparado también con predecir un delta cero.
No modifica el checkpoint de origen ni el entrenamiento en ejecución. Esta prueba
de primeras transiciones no evalúa la memoria de trayectorias largas.

Ejecuta CUDA compilada, BF16 y AdamW de 8 bits. Informa error de delta y de píxel
en los ejemplos usados para ajustar (`fit`) y en otras semillas (`unseen`). Un
buen ajuste del conjunto fijo no prueba generalización ni habilita avanzar de
nivel. No guarda observaciones ni checkpoints; el entrenamiento normal mantiene
sus validaciones separadas y sus umbrales completos.

El entrenamiento normal también admite una etapa explícita con representación
congelada, cuando un checkpoint ya reconstruye bien:

```sh
.venv/bin/python -m model train --checkpoint model/runs/entrenamiento/latest.pt \
  --freeze-representation --output model/runs/predictor --max-steps 20000
```

Esta etapa sigue recogiendo partidas reales y muestreando replay, reconstruyendo
los prefijos con el encoder en cada actualización. No almacena latentes aprendidos.
Solo cambia los parámetros del predictor; conserva los pesos y momentos del
optimizador del encoder y decoder. El gradiente visual atraviesa el decoder
congelado para enseñar al predictor. Omite las pérdidas de reconstrucción y la
regularización del encoder, que no podrían actualizarlo; el panel las muestra
como no disponibles, no como errores cero. La reconstrucción y todos los umbrales
se siguen evaluando normalmente. El modo se conserva al reanudar y
`--no-freeze-representation` restablece el aprendizaje conjunto. Sin la política
automática, superar un nivel no cambia este modo: otros juegos pueden requerir
adaptar la representación. La CLI exige un checkpoint para comenzar congelado, pero no presupone que sea bueno;
hay que comprobar su reconstrucción en validación.

Para alternar automáticamente entre ambos modos:

```sh
.venv/bin/python -m model train --checkpoint model/runs/entrenamiento/latest.pt \
  --auto-representation --representation-unfreeze-error .02 \
  --representation-freeze-error .01 --representation-gate-patience 2 \
  --output model/runs/entrenamiento
```

El gate usa el **error de delta reconstruido desde el latente real**: mide al
encoder y decoder juntos. El error de predicción no lo controla; el predictor
puede necesitar más aprendizaje aunque la representación ya sea precisa.
Después de cada validación congelada, si algún nivel evaluado supera el 2% en
dos validaciones consecutivas, se reactivan encoder y decoder. Si todos están
en el 1% o menos, con diversidad latente suficiente, durante dos validaciones,
se congelan de nuevo. El predictor se entrena en ambos modos y el gradiente de
su error visual sigue atravesando el decoder congelado.

Los niveles deben cumplir las coberturas mínimas de transiciones y cambios.
Datos insuficientes/no finitos o un valor intermedio rompen la racha; nunca se
interpretan como error cero. Un cambio de niveles evaluados reinicia las rachas.
La política está desactivada por defecto para conservar el comportamiento de
checkpoints anteriores. `--no-auto-representation` la desactiva manteniendo el
modo actual, y los cambios explícitos de política/modo reinician sus rachas.

Los cambios se aplican entre actualizaciones, conservando parámetros y momentos
de ambos optimizadores. Guardan `latest.pt` inmediatamente y reinician la racha
de aprobación del currículo, sin cambiar sus umbrales. El checkpoint conserva
el modo y las rachas parciales del gate en una reanudación normal. CUDA continúa
usando `torch.compile`; la primera transición de modo puede compilar otra
especialización. El panel muestra el modo actual, el gate y su última evidencia.

## Currículo y juegos del SDK

Por defecto, `sc01-v1` recorre los niveles 1–5. En cada GAME_OVER, victoria o
límite de 256 acciones, crea un episodio del mismo nivel con una semilla nueva.
Solo avanza de nivel tras tres validaciones consecutivas satisfactorias,
comprobando también los niveles anteriores. Conserva todos los pesos al avanzar.

**El SDK 0.9.9 no ofrece selección portátil de nivel.** `environment.py` mantiene
una capacidad explícita y acotada para aislar los niveles de los juegos sintéticos
incluidos (`sc01-v1`, `si01-v1`) con arcengine 0.9.3. Usa el motor local para
seleccionar nivel y terminar al resolverlo, sin añadir el tablero aleatorio del
siguiente nivel a la predicción. No cambia reglas, posiciones, acciones ni frames
de ese nivel. El learner nunca recibe el motor ni sus reglas internas.

Cualquier otro juego instalado en el SDK usa su progresión nativa:

```sh
.venv/bin/python -m model train --game GAME_ID --environments-dir /ruta/a/juegos --native --output model/runs/otro-juego
.venv/bin/python -m model train --game si01-v1 --levels 1 2 3 4 5 --output model/runs/interacciones
```

En la progresión nativa, un cambio no terminal de `levels_completed` separa los
historiales. La acción cuenta como interacción real, pero el tablero inicial del
nuevo nivel no es un objetivo de predicción del anterior. El runner conserva el
mismo juego del SDK, libera el replay previo e inicia otro episodio con memoria
cero y **solo el último frame** de la respuesta: los anteriores pueden contener
la animación ganadora del nivel anterior. El learner rechaza prefijos que mezclen
ambos niveles. Los sucesores terminales WIN y GAME_OVER siguen siendo objetivos
válidos, incluido el WIN de un nivel sintético aislado.

`--mode normal` permite el modo normal del SDK; por defecto se usa offline.
Este runner es entrenamiento local, no una entrega de competición. No utiliza
datos privados ni incorpora un solucionador del juego. En juegos arbitrarios,
los umbrales dependen de que la dinámica sea predecible desde las observaciones.

### Continuación automática de la práctica

`model.sequence` continúa un checkpoint de Square Cross por sus **5 niveles**,
después por los **5 de Square Cross Interactions** y finalmente por los juegos
públicos de entrenamiento del SDK. Conserva pesos, optimizadores y configuración;
al cambiar de juego vacía replay, memoria recurrente, contadores y rachas del gate.
El modo actual del gate se conserva y las siguientes validaciones lo adaptan.
Una mejora del predictor no se presenta como una victoria en el juego.

Desde un checkout limpio con la implementación confirmada:

```sh
.venv/bin/python -m model.sequence plan \
  --plan model/runs/active-training/sequence.json \
  --sdk-environments-dir environment_files \
  --refresh-checkout /ruta/a/arc3-training-checkout \
  --refresh-branch training/practice-sequence
# Detener antes cualquier trainer que escriba en este mismo output.
.venv/bin/python -u -m model.sequence run \
  --plan model/runs/active-training/sequence.json \
  --output model/runs/active-training --device cuda
```

El plan congela el catálogo público del SDK. Según
[la política de entrenamiento](../docs/training-policy.md), ordena los 25 IDs
base y reserva cinco juegos completos usando `Random(42)`; los otros 20 se
recorren en ese orden, reservando también el último nivel de cada uno. Un catálogo
incompleto o con versiones duplicadas exige revisar el plan antes de continuar.
El plan y la etapa residen juntos en el checkpoint: un reinicio no repite una
transferencia. El supervisor bloquea un segundo supervisor del mismo output.
El trainer hereda ese bloqueo y lo conserva aunque el supervisor desaparezca:
un reinicio se rechaza mientras el trainer anterior siga vivo. SIGINT o SIGTERM
al supervisor pide al trainer guardar el checkpoint y espera a que termine;
las señales repetidas no interrumpen ese guardado. Tras una muerte forzada del
supervisor, detener el trainer anterior con SIGINT y esperar su salida antes de
reanudar la secuencia.

Tras completar Square Cross, crea una rama **local privada** en la ruta indicada,
actualiza `origin/main` y hace rebase allí. No reescribe ni publica la rama de la
PR. Un checkout sucio, conflicto o fallo del entrenador detiene la secuencia
conservando el último checkpoint. Solo `curriculum_converged` con todo el currículo
completado permite cambiar de juego; una interrupción, colapso o límite no lo hace.
Para reanudar basta repetir `sequence run`. No se aplica `--stop-after-level`.

En la fase SDK, la respuesta que entra al nivel reservado se descarta **antes**
de codificarla o incorporarla al replay, pérdidas o métricas; se inicia otra
partida. La validación exige los umbrales y cobertura mínima para **cada nivel
permitido**, además del agregado, y registra `reserved_level_boundaries` y
`native_levels`. La política aleatoria podría no alcanzar todos esos niveles;
en ese caso la secuencia permanece en el juego actual. Nunca salta a evaluación.

## Validación, avance y progreso

Cada 1.000 acciones se ejecutan 8 episodios de hasta 128 acciones con pesos
congelados. Las semillas de entrenamiento son pares; las de validación son
impares, fijas y separadas. La validación no escribe replay ni actualiza pesos.
Incluye la primera acción, los cambios del HUD y las transiciones terminales.
En modo nativo, también reinicia la memoria desde el último frame al avanzar de
nivel y excluye ese cruce de los errores predictivo y de reconstrucción. Reporta
esas acciones como `level_advances`, separadas de `skipped` (datos no evaluables);
las transiciones mínimas y los umbrales del currículo conservan su significado.

Para avanzar, **cada nivel evaluado** debe cumplir:

- Al menos 128 transiciones evaluables y 32 con cambios reales.
- Error del delta visual ≤ 1%: celdas incorrectas / unión de cambios reales y
  predichos. Penaliza cambios omitidos, inventados y colores erróneos.
- Error del delta reconstruido desde el latente real ≤ 1%, con la misma
  cobertura mínima. Acertar el predictor no basta si falla la reconstrucción.
- MSE del delta latente ≤ .01 y desviación típica media del latente ≥ .1, para
  impedir que un encoder colapsado apruebe por producir siempre el mismo vector.
- Ninguna transición omitida por falta de frames o incompatibilidad geométrica.

Son valores iniciales configurables, no resultados de convergencia demostrados.
Los eventos JSON de terminal muestran acciones, actualizaciones, pérdidas,
errores, partidas, acciones/s, memoria GPU máxima y checkpoints. Victorias y
GAME_OVER se reportan aparte de la calidad predictiva.
`delta_error` mide la predicción causal; `decoder_delta_error` mide la
reconstrucción desde el latente real posterior. La segunda se calcula después
de puntuar la primera y nunca se utiliza para corregirla. `latent_std` mide
variación entre observaciones de la misma geometría y posición, sin mezclar
posiciones o tamaños distintos para aparentar diversidad.
`prediction_quality` y `reconstruction_quality` añaden errores promediados por
color, separados entre píxeles estáticos y cambiados, junto con recuentos y
errores por color. Ayudan a detectar detalles olvidados aunque el fondo domine
la imagen. Son diagnósticos; no sustituyen el error de delta ni los umbrales.

Se detiene guardando `latest.pt` al llegar a 100.000 acciones sin converger en
un nivel, a 500.000 acciones totales, al completar el currículo o con Ctrl+C.
Los límites se ajustan mediante `--max-level-steps` y `--max-steps`.
El valor `0` desactiva ese límite. Para una práctica hasta convergencia del nivel
actual o colapso sostenido:

```sh
.venv/bin/python -m model train --levels 1 --max-steps 0 --max-level-steps 0 --stop-after-level --output model/runs/practice
```

Se considera colapso sostenido `latent_std < .01` en tres validaciones seguidas,
después de haber observado diversidad suficiente (`>= min_latent_std`) o de
256 actualizaciones de arranque. Los parámetros `--collapse-std`,
`--collapse-patience` y `--collapse-after-updates` controlan este detector.
El final y los checkpoints indican `encoder_collapsed`, `level_converged` o
el límite/interrupción correspondiente. Una varianza alta por sí sola no
garantiza representaciones útiles: también deben bajar los errores visuales.

## Panel web en directo

Para mostrar también goal y planner, pasa ambos logs independientes:

```sh
.venv/bin/python -m model.dashboard --log model/runs/current.log \
  --behavior-log model/runs/goal-planner/current.log
```

Un único selector **Entrenamiento** gobierna cabecera, contadores, gráficas y tablas.
**En curso** sigue automáticamente la fuente que escribió más recientemente y,
en una campaña, su variante y semilla actuales. Una selección explícita queda fija
aunque el proceso pase a otra variante. Las pestañas **Predictor y representación**,
**Planner y actor** y **Evaluaciones** comparten esa selección; no tienen selectores
independientes de nivel, fase o ejecución. Las series/tablas identifican fase,
política y juego/nivel; una métrica ausente nunca se rellena desde otro entrenamiento.

Se pueden consultar las ejecuciones disponibles en los logs configurados, incluidas
las variantes/semillas anteriores de la campaña. Se conservan hasta cien resúmenes
de campaña en memoria, reconstruibles desde el log; las curvas siguen limitadas
a la ventana de puntos disponible. No es un archivo permanente de todos los
entrenamientos: si el log se reemplaza o se lee sólo su tramo final, el panel lo
indica y no sustituye silenciosamente una selección desaparecida por la activa.

La pestaña de comportamiento muestra recogida de acciones/partidas por ronda,
época y actualizaciones, pérdidas de eventos (CE), V y Q (BCE), victorias frente
al azar, intervalo de Wilson, acciones por victoria, ms de decisión y acciones/s.
Las pérdidas no son porcentajes de error. Desarrollo, evaluación final y evaluación
aislada tienen series y filas identificadas; sus acciones nunca se suman al contador del mundo ni a la
recogida. Los nuevos eventos incluyen ronda y fase, y se actualizan al completar
cada bloque de 16 partidas o cada época. El estado terminado sigue siendo
terminado aunque el log envejezca. Cada log reinicia su ventana por separado.

Logs previos al desglose conservan solo sus métricas registradas: sin inventar
componentes de pérdida ni fase de evaluación. Se muestran bajo «fase no
registrada». El panel no reanuda procesos ni crea un historial adicional.

### Exportar el goal y el plan al viewer

```sh
.venv/bin/python -m model.behavior_trace \
  --checkpoint model/runs/active-training/latest.pt \
  --heads model/runs/goal-planner/more-experience/heads.pt \
  --seed 9200517 --level 2 \
  --output model/runs/goal-planner/new-plan.arc3
```

Ejecuta una partida de inspección con pesos congelados, sin aprendizaje, y exige
una ruta nueva. El archivo 1.1 conserva observaciones, acciones y el delta visual
predicho; añade `planning` con la rama elegida **antes** de ejecutar. El viewer
muestra V, score, probabilidades de progreso/GAME_OVER y cada acción del plan.
Solo se ejecuta la primera, después se observa y replantea. No se inventa una
imagen objetivo. `Q` muestra una sola elección sin trayectoria imaginada; el
horizonte MPC se lee del checkpoint de comportamiento. `--device cpu` permite
inspección sin CUDA. La CLI sigue limitada a los dos juegos sintéticos admitidos.

La ruta opt-in `--goal-planning` sí propone una **hipótesis de objetivo visual**:

```sh
.venv/bin/python -m model.behavior_trace \
  --checkpoint /usr/local/src/arc-agi-3-model/model/runs/active-training/latest.pt \
  --heads /usr/local/src/arc-agi-3-model/model/runs/goal-planner/multilevel/heads.pt \
  --goal-planning --goal-horizon 8 --goal-beam 4 --goal-lifetime 8 \
  --plan-horizon 4 --plan-beam 4 --seed 9426301 --level 3 --steps 128 \
  --output model/runs/issue52/goal-level3.arc3
```

`goal_planning.py` primero busca desde el estado real causal con el readout SDK
congelado y guarda el mejor sucesor entre profundidades 1..8. Después, una búsqueda
independiente de horizonte 4 elige la rama final por `exp(-MSE(z, goal_z))`, con
MSE en FP32 sobre todas las entradas; V no se mezcla en ese ranking. Conserva el
mismo tensor objetivo durante 8 acciones reales, observa y replantea. `reset()`
limpia el objetivo al comenzar otra partida; `after_real_action()` avanza su edad
y limpia en RESET, frontera SDK o terminación. También se renueva al cambiar la
geometría. La proximidad nunca declara victoria.

El decoder recibe el objetivo exacto y los sucesores retenidos de la rama elegida,
antes de la acción. Las imágenes son diagnósticos sin influencia en el ranking:
objetivo y último frame del plan siguen separados. No hay red nueva, entrenamiento,
optimizador ni replay. La CLI compara ambos fingerprints antes/después. CUDA usa
Engine con compile/BF16; `--device cpu` permite redes pequeñas. Horizontes, beams
y clicks admiten 1..64; vida 1..1024. Los defaults y pesos de q/mpc siguen vigentes
sin el flag. Esto es un diseño experimental adoptado, no una reproducción de un
paper ni una solución garantizada al descubrimiento de objetivos desconocidos.

Pruebas funcionales y fixture de red pequeña en CPU (sin suite visual):

```sh
PYTHONPATH=src:tests .venv/bin/python -m unittest model.tests.test_planning model.tests.test_goal_planning tests/test_arc3_codec.py
.venv/bin/python viewer/tests/generate_fixtures.py --goal-network
cd viewer
npm test
npm run build
```

Inspección congelada del 2026-10-14: una partida por nivel, seeds fijados antes
de exportar, presupuesto 128, propuesta 8/4, plan 4/4 y vida 8. Sin selección de
victorias ni reentrenamiento. El viewer publicado conserva los ejemplos previos.

| Nivel / viewer | Seed | Acciones | Terminación SDK | Wall s | Mediana decisión ms |
| --- | --- | --- | --- | --- | --- |
| [3](https://murillo128.hopto.org:8443/arc3-viewer/?file=https%3A%2F%2Fmurillo128.hopto.org%3A8443%2Farc3-viewer%2Fexamples%2Fgoal-level3.arc3) | 9426301 | 43 | GAME_OVER | 52.641 | 91.409 |
| [4](https://murillo128.hopto.org:8443/arc3-viewer/?file=https%3A%2F%2Fmurillo128.hopto.org%3A8443%2Farc3-viewer%2Fexamples%2Fgoal-level4.arc3) | 9426401 | 108 | GAME_OVER | 54.992 | 89.551 |
| [5](https://murillo128.hopto.org:8443/arc3-viewer/?file=https%3A%2F%2Fmurillo128.hopto.org%3A8443%2Farc3-viewer%2Fexamples%2Fgoal-level5.arc3) | 9426501 | 33 | GAME_OVER | 45.635 | 93.095 |

Las tres exportaciones sumaron 153.268 s de tiempo de partida registrado,
incluyendo inicialización/compile; no son un benchmark de latencia. Las máximas
decisiones fueron 28.795, 23.630 y 23.157 s. GPU mantuvo el límite de 100 W y
no se alteraron otros procesos. Ambos fingerprints fueron idénticos antes/después:

- Mundo: `72b2f1dbbf74bc0c4faf02ae16a7dcb50fb04696523bfa7f3c04d3f1138248ad`.
- Readout: `3061a1f100893f4587881df35554751d90cbaf0778bb22d6ac5c3f41c945611a`.

Archivos ignorados en `model/runs/issue52/`; tamaños 13.185, 29.002 y 14.620 bytes.
SHA-256 (nivel 3/4/5):

```text
82b4ebbb08cf7e1738244345ed8a2769c5a95f12591f3eb13e7c0338c3af23b6
c75c128e80c70b57260fafd6d5f1bed0875e68578c31221e47d20bc60983efda
09377228053e09dcb96f01ed043b99716194e0f63af082715ad97864a1f5aee4
```

Codec verificó modo/objetivo, acciones y geometría; la creación del objetivo
se renovó en posiciones 0,8,16… y sus bytes se mantuvieron durante cada vida.
Objetivo y endpoint difirieron visualmente en 43/43, 107/108 y 33/33 decisiones.
Descargas HTTPS de los tres archivos y assets (HTML, JS/CSS, worker/WASM) coincidieron
byte por byte con la compilación y los archivos locales. La inferencia y las
imágenes siguen siendo hipótesis: las tres partidas fallaron y no establecen una
mejora de éxito ni generalización.

Validación: 42 pruebas de planning/goal/codec, 27 pruebas de codec/fixtures,
223 pruebas Vitest y build/type-check. Fixture CPU de red real conserva imágenes,
acciones y distancias al pasar por Python y TypeScript. El validador existente
de LMEX acepta el grafo con 1.739 nodos / 2.968 edges y cobertura parcial previa.
No se ejecutaron pruebas visuales/browser.

El primer sample de esta extensión, seed 9200517, tuvo 24 acciones y GAME_OVER
con el MPC de horizonte 1 guardado; se verificaron intactos pesos del mundo y
del evaluador. Es una muestra de inspección, no una evaluación de rendimiento.

### Métricas del mundo

El panel lee los eventos JSON del log existente, sin cargar el modelo, usar GPU,
interrumpir el entrenamiento ni guardar otra copia del historial:

```sh
.venv/bin/python -m model.dashboard --log model/runs/exploration/run.log
```

Abre `http://127.0.0.1:8766/`. `--host` y `--port` cambian la dirección de escucha;
por defecto solo escucha en loopback. Para un entrenamiento nuevo que aún no
guarde la salida, redirígela a un log (crea primero el directorio):

```sh
mkdir -p model/runs/practice
.venv/bin/python -u -m model train --output model/runs/practice > model/runs/practice/run.log 2>&1
```

Consulta cada 2 segundos y muestra acciones, partidas iniciadas, actualizaciones,
nivel, acciones/s, pico de memoria asignada por PyTorch y gráficas de pérdidas,
errores de delta visual y latente, y diversidad del latente. Cada arranque,
también al reanudar, establece un **punto limpio al terminar la primera acción y
su actualización**. El panel cuenta desde ese instante las acciones, actualizaciones,
partidas nuevas y acciones del nivel; las gráficas y marcas de validación/checkpoint
usan el mismo origen. Antes de ese punto muestra que está preparando la medición;
en el primer punto los contadores son cero y aún no hay TPS. Los contadores globales
del checkpoint y las cadencias de aprendizaje/validación permanecen intactos.
También reinicia la medición del pico de memoria GPU en ese punto.

TPS significa **acciones por segundo desde el punto limpio**: excluye la carga y
la primera acción, pero incluye validación, checkpoints y cualquier compilación
posterior. No son tokens/s ni una medida instantánea. Los errores de predictor
y decoder se actualizan cuando termina cada validación, a la cadencia configurada.
Las acciones de validación no se suman al contador de entrenamiento. Las series
separadas por nivel y fase evitan unir evaluaciones distintas en una curva.

El eje horizontal usa acciones reales. El panel mantiene hasta 2.000 puntos por
tipo en RAM (`--max-points`) y al abrir un log grande lee los últimos 8 MiB.
No crea base de datos ni archivos de métricas. Reconoce truncado, sustitución del
log y nuevos arranques; elimina entonces las curvas de la ejecución anterior.
Si faltan datos recientes o el log desaparece, lo indica y reintenta; no confunde
la disponibilidad del servidor web con la actividad del trainer. Cada registro de
progreso contiene estadísticas completas desde el punto limpio: reiniciar solo
el panel o leer desde la mitad del log conserva ese origen. Los logs anteriores
se identifican como acumulados y no se les inventa una duración de calentamiento.

Para servirlo bajo el HTTPS existente, configura un proxy a su puerto local,
conservando la barra final que elimina el prefijo al reenviar:

```nginx
location = /arc3-training { return 301 /arc3-training/; }
location ^~ /arc3-training/ {
    proxy_pass http://127.0.0.1:8766/;
    proxy_buffering off;
}
```

Solo expone los assets del panel y `/api/state`; no sirve logs, checkpoints,
archivos arbitrarios ni controles para modificar el entrenamiento. No requiere
dependencias web, CDN ni compilación de frontend.

## Checkpoints y validación aislada

```sh
.venv/bin/python -m model train --checkpoint model/runs/square-cross/latest.pt --max-steps 1000000
.venv/bin/python -m model validate --checkpoint model/runs/square-cross/latest.pt
.venv/bin/python -m model validate --checkpoint model/runs/square-cross/latest.pt --levels 1 2 3 --validation-episodes 16
```

`latest.pt` se reemplaza atómicamente cada 1.000 acciones y al detenerse;
`best.pt` conserva la mejor evaluación visual del nivel actual. Guardan pesos,
ambos optimizadores, configuración, contadores, currículo y RNG. Al reanudar,
empieza una partida nueva con la caché de entradas vacía: recupera el entrenamiento, no la
misma trayectoria a mitad de partida. Los argumentos de entrenamiento dados
explícitamente sustituyen los guardados y reinician las rachas de aprobación.
`--output` permite ubicar checkpoints fuera del repositorio.
El checkpoint incluye FP8 y la precisión del optimizador. Reanudar con la misma
precisión restaura los momentos. Cambiar entre AdamW de 32 y 8 bits exige
`--reset-optimizer`: conserva pesos, contadores y RNG, pero reinicia los momentos.
La validación CPU puede leer los pesos FP32 de un entrenamiento FP8 sin necesitar
los paquetes CUDA adicionales. Al migrar una ejecución antigua al aprendizaje por
jugada, usar `--train-every 1 --updates-per-train 1 --loss-steps 1 --min-observations 2`.
Los checkpoints anteriores al estado espacial no son compatibles con esta
arquitectura: se empieza un entrenamiento nuevo, sin adaptar pesos silenciosamente.

## Sincronización con LLM Model Explorer

LMEX admite pesos Safetensors y solo sigue enlaces que permanezcan dentro de su
raíz de modelos. Un symlink directo al repo o a `latest.pt` no sirve: el checkpoint
contiene pesos, optimizadores y RNG, en un formato distinto. Este comando publica
solo los pesos guardados y la arquitectura actual:

```sh
.venv/bin/python -m model.lmex_sync \
  --checkpoint model/runs/exploration/latest.pt \
  --model-root /home/sergio/.cache/llm-model-explorer/models \
  --watch
```

Comprueba cambios cada 5 segundos (`--interval`). Sin `--watch`, publica una vez.
`--architecture` permite elegir otro JSON y `--model-id` otro nombre de entrada.
La primera publicación exige una entrada libre: si ya existe una carpeta antigua
con ese nombre, muévela fuera de la raíz de modelos antes de ejecutar el comando.
El sincronizador nunca reemplaza una carpeta o enlace ajeno silenciosamente.

Copia `architecture.json` byte por byte y exporta `weights` de `latest.pt` en CPU
con los mismos nombres, formas, tipos y valores. En checkpoints de secuencias
incluye los pesos de `ChangeHead` con el prefijo `exploration.`, sin su replay ni
optimizador, y conserva el contador de batches. No carga optimizadores en GPU ni
ejecuta el modelo. Construye el paquete completo dentro de la raíz de LMEX y
cambia atómicamente un enlace interno. Conserva solo el paquete activo; si cambia
solo la arquitectura, reutiliza el archivo inmutable de pesos sin otra exportación.
Un error o un checkpoint sustituido durante la conversión conserva el paquete
anterior y se reintenta. El entrenamiento sigue independiente.

Los pesos de LMEX son los del **último checkpoint guardado**, no los parámetros
en RAM tras cada actualización. La configuración exportada incluye los contadores
de ese checkpoint. El refresco normal de LMEX detecta el cambio sin reiniciar su
backend. La arquitectura describe el perfil de secuencias de `subgoal_config.json`:
una invocación compartida del encoder, memoria recurrente, predictor global/local,
decoder y cabeza de exploración. Sus 497 tensores tienen bindings exactos; no
requiere los pesos del antiguo `PathValue`, ausente en este entrenamiento. La
cobertura parcial se refiere al detalle de scheduling, pérdidas y optimizadores.
Los pesos reales también se pueden inspeccionar en Tensor Explorer. Otros perfiles
necesitan su propia descripción mediante `--architecture`. Los paquetes exportados,
pesos y logs quedan fuera de Git.

Al cambiar el entrenamiento activo, actualiza tanto `--checkpoint` como la revisión
de código/arquitectura del servicio de sincronización. El trainer no necesita
reiniciarse. Antes de publicar un perfil nuevo, valida un paquete materializado
con el backend instalado del Explorer:

```sh
python -m llm_model_explorer.validate_architecture_cli /path/to/model-package --json
```

Comprueba además que una sesión nueva del endpoint público devuelve la arquitectura
disponible y sus recursos de tensores. El refresco del selector de modelos abre el
checkpoint nuevo; las sesiones fijadas al paquete anterior no se convierten
silenciosamente en el nuevo.

## Comprobaciones locales

```sh
.venv/bin/python -m unittest discover -s model/tests -v
```

Comprueban la aritmética/probabilidades del delta, el gradiente visual conjunto,
la equivalencia entre microbatches, la diversidad entre observaciones, la reanudación,
la validación congelada y las fronteras reales del SDK. No ejecutan navegador.
`architecture.json` conserva los IDs de los componentes neuronales compartidos y
describe el perfil de secuencias actual. La sincronización de LMEX usa los pesos
reales del checkpoint, no los pesos aleatorios de la antigua maqueta.

Las pérdidas se inspiran en [SIGReg de LeJEPA](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md)
y en la [restricción de varianza de VICReg](https://github.com/facebookresearch/vicreg).
La combinación temporal, el muestreo de posiciones y sus pesos son decisiones
de esta implementación, no una reproducción de sus resultados publicados.
