# Modelo residual latente

Primera implementación del encoder, predictor recurrente y decoder visual. Se
entrenan con acciones aleatorias legales y transiciones reales del SDK. El
actor Dreamer, el crítico distribucional y el entrenamiento de políticas por
imaginación permanecen como componentes futuros en
[`architecture.json`](architecture.json). El experimento separado
`model.behavior_experiment` implementa un primer evaluador de objetivos/valor y
un planner latente sobre pesos congelados; se describe a continuación.

El código y la arquitectura viven en este directorio. No modifica los comandos
existentes `training.collect`, `evaluation.run`, el viewer ni la exportación de
trazas. Los checkpoints y el replay no se suben a Git.

El diagrama refleja la variante en experimentación con `--pixel-latent`,
`--broadcast-conditioning`, `--local-dynamics`, y escalas residual/posicional .1.
Sus operaciones enlazan los 491 tensores reales compartidos entre llamadas. La
configuración del checkpoint identifica qué variante ejecuta cada entrenamiento;
el diagrama no demuestra convergencia ni generalización entre juegos.

## Objetivo y planner con el modelo del mundo congelado

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
los readouts. La primera ronda recoge al azar. Desarrollo y evaluación final
usan semillas distintas del entrenamiento. La evaluación final se hace una
vez tras seleccionar el candidato; no entrena ni guarda replay.

`report.json` contiene recuentos, semillas, acciones, tiempos e intervalos de
Wilson; `heads.pt` solo añade pesos de comportamiento, la huella del mundo y
configuración. El replay de latentes reales es FIFO en RAM, máximo 8192
transiciones por defecto (aproximadamente 4 GiB para este modelo), y desaparece
al salir. La memoria recurrente `h` se reinicia por partida. Se imprimen eventos
JSON de recogida, ajuste y evaluación. No escribe al dashboard del trainer del
mundo ni arranca automáticamente el currículo anterior.

Para evaluar esos readouts sin entrenamiento adicional:

```sh
.venv/bin/python -u -m model.behavior_experiment \
  --checkpoint model/runs/active-training/latest.pt \
  --heads model/runs/goal-planner/new-run/heads.pt \
  --output model/runs/goal-planner/evaluation \
  --game sc01-v1 --level 2 --device cuda --seed-offset 20000
```

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
   Añade un término diferenciable de error sobre la unión de deltas: si `c` es
   el número de celdas realmente cambiadas, `f` la suma de probabilidades de
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
Las acciones de validación no se suman al contador de entrenamiento. El selector
de nivel evita mezclar validaciones de niveles distintos en una curva.

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
con los mismos nombres, formas, tipos y valores. No carga optimizadores en GPU ni
ejecuta el modelo. Construye el paquete completo dentro de la raíz de LMEX y
cambia atómicamente un enlace interno. Conserva solo el paquete activo; si cambia
solo la arquitectura, reutiliza el archivo inmutable de pesos sin otra exportación.
Un error o un checkpoint sustituido durante la conversión conserva el paquete
anterior y se reintenta. El entrenamiento sigue independiente.

Los pesos de LMEX son los del **último checkpoint guardado**, no los parámetros
en RAM tras cada actualización. La configuración exportada incluye los contadores
de ese checkpoint. El refresco normal de LMEX detecta el cambio sin reiniciar su
backend. La arquitectura sigue declarando cobertura parcial: no se inventan
bindings entre sus cajas y los tensores; los pesos reales se pueden inspeccionar
en Tensor Explorer. Los paquetes exportados, pesos y logs quedan fuera de Git.

## Comprobaciones locales

```sh
.venv/bin/python -m unittest discover -s model/tests -v
```

Comprueban la aritmética/probabilidades del delta, el gradiente visual conjunto,
la equivalencia entre microbatches, la diversidad entre observaciones, la reanudación,
la validación congelada y las fronteras reales del SDK. No ejecutan navegador.
`architecture.json` conserva los IDs del diseño importado y describe el perfil
base. Declara cobertura parcial para los componentes futuros y para los bindings
de parámetros aún pendientes. La sincronización de LMEX usa los pesos reales del
checkpoint, no los pesos aleatorios de la antigua maqueta.

Las pérdidas se inspiran en [SIGReg de LeJEPA](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md)
y en la [restricción de varianza de VICReg](https://github.com/facebookresearch/vicreg).
La combinación temporal, el muestreo de posiciones y sus pesos son decisiones
de esta implementación, no una reproducción de sus resultados publicados.
