# Modelo residual latente

Primera implementación del encoder, predictor recurrente y decoder visual. Se
entrenan con acciones aleatorias legales y transiciones reales del SDK. El
actor aprendido, el crítico, la recompensa, la imaginación para entrenar políticas
y las cabezas auxiliares permanecen como componentes futuros en
[`architecture.json`](architecture.json); **no se ejecutan ni se entrenan aquí**.

El código y la arquitectura viven en este directorio. No modifica los comandos
existentes `training.collect`, `evaluation.run`, el viewer ni la exportación de
trazas. Los checkpoints y el replay no se suben a Git.

## Instalación y ejecución

Desde la raíz del repositorio, con Python 3.12:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python -r model/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m model train
```

Se elige CUDA cuando está disponible y se activa `torch.compile`, SDPA, BF16
cuando el dispositivo lo soporta, AdamW fusionado y checkpointing de activaciones.
La primera actualización incluye el coste de compilación. `--no-compile` permite
ejecución eager; una compilación fallida no se oculta como un cambio automático
de modo. El rendimiento depende del hardware y de las longitudes de las partidas.
No se afirma que compile siempre sea más rápido; evita recompilar usando la misma
configuración. Se registra el tiempo total, incluido el arranque, en el progreso.

El modelo **base** tiene 214.143.104 parámetros:

| Componente | Configuración | Parámetros |
|---|---|---:|
| Visual encoder | 12 bloques, ancho 768, 12 cabezas; memoria de animación 1024 por patch | 91.525.376 |
| Dynamics | Memoria global 2048; predictor de 6 bloques, ancho 768; latente P×1024 | 64.929.152 |
| Visual decoder | 8 bloques, ancho 768, 12 cabezas; 16 colores | 57.688.576 |

`--profile small --device cpu --no-compile` usa la misma computación con
74.872 parámetros para desarrollo. No es un sustituto de capacidad equivalente.
`--frame-chunk` limita frames por llamada espacial; `--batch-episodes` controla
cuántas partidas se muestrean como mínimo. Por defecto mezcla 4 partidas y
exige 64 observaciones reales, con microbatches de 4 frames. El perfil base no se reduce
automáticamente al encontrar límites de memoria.
La compilación puede aumentar el pico de memoria. En una GPU compartida de
12 GiB, la mezcla de partidas puede requerir `--no-compile`; mantiene BF16 y
SDPA. No hay fallback silencioso ni se reduce el tamaño de la red.

## Flujo y entrenamiento

1. El adaptador abre un episodio con semilla nueva y obtiene la observación SDK.
   No añade una acción artificial a la primera observación.
2. El comportamiento aleatorio elige una acción de `available_actions`; ACTION6
   incluye coordenadas SDK uniformes en 0…63. Guarda el resultado real en RAM.
3. Cada 32 acciones, muestrea partidas y vuelve a codificar sus **prefijos
   completos** con los pesos actuales. Conserva todos los frames de animación.
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
   + 10 * variance_floor`, además de los gradientes visuales del siguiente paso.
   `--sigreg-weight` controla `w`; los pesos activos se imprimen al iniciar.
   Ambos latentes reales conservan gradiente. SIGReg usa al menos 64
   observaciones reales distintas, proyecciones aleatorias y cálculo FP32.
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
   no cambiadas. Dentro de cada grupo, pondera cada píxel por la inversa de la
   raíz cuadrada de la frecuencia de su **color real**, normalizando los pesos.
   Así los detalles estáticos pequeños ganan peso sin darles tanta masa como
   a todo el fondo; no se identifican objetos ni colores especiales del juego.
   `--color-balance-power` controla el exponente (0: uniforme por píxel,
   1: igual masa por color; por defecto .5). `--visual-balance pixel` selecciona
   explícitamente la ponderación original; el valor por defecto es `color`.
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

El objetivo total suma los términos latentes y las tres pérdidas visuales
(NLL más error de unión suave por rama). AdamW usa `lr=1e-4` por
defecto. Los dos optimizadores conservan conjuntos de parámetros disjuntos,
pero todas las pérdidas se calculan con los mismos pesos previos a la actualización.
Para limitar memoria, cada microbatch del decoder acumula gradientes en una hoja
temporal; al terminar se propagan explícitamente por el grafo original de ambos
latentes. El encoder conserva las salidas de cada episodio y recalcula sus
activaciones internas durante backward, para poder mezclar más partidas sin
retener todos sus grafos internos a la vez. Ese puente **no corta el gradiente**.
Se comprueban y recortan ambos
conjuntos de gradientes antes de actualizar ningún peso.

Los píxeles del tablero anterior no entran a la red del decoder: solo determinan
la geometría y la interpretación de sus colores como deltas. La suma residual
opera por patch y sigue permitiendo imaginar sin generar píxeles.

Los estados recurrentes de validación avanzan con observaciones reales. El
entrenamiento aleatorio no calcula inferencias en vivo que no va a consumir: las
reconstruye en los batches. `Engine.rebuild` permite reconstruir un estado en vivo
después de actualizar los pesos. No existe caché persistente de latentes obsoletos.

El replay es una cola limitada a 32 episodios / 128 MiB, con observaciones nativas
uint8, inicialización y acciones. Se entrenan hasta las últimas 16 transiciones
de cada prefijo; los gradientes atraviesan el prefijo completo. Si las partidas son
cortas se combinan varias hasta reunir las observaciones mínimas, sin mezclar
sus memorias. Los episodios anteriores permanecen hasta su expulsión por capacidad.
Una respuesta sin frames se conserva, avanza la memoria con la acción cuando
hay estado y no inventa un objetivo visual. El encoder admite geometrías nativas
rectangulares y padding interno; predecir cambios de tamaño de canvas queda fuera
de este primer decoder. Tales transiciones se omiten del loss y hacen fallar el
criterio de validación, en lugar de contarlas como aciertos.

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
.venv/bin/python -m model train --game si01-v1 --levels 1 2 3 --output model/runs/interacciones
```

`--mode normal` permite el modo normal del SDK; por defecto se usa offline.
Este runner es entrenamiento local, no una entrega de competición. No utiliza
datos privados ni incorpora un solucionador del juego. En juegos arbitrarios,
los umbrales dependen de que la dinámica sea predecible desde las observaciones.

## Validación, avance y progreso

Cada 1.000 acciones se ejecutan 8 episodios de hasta 128 acciones con pesos
congelados. Las semillas de entrenamiento son pares; las de validación son
impares, fijas y separadas. La validación no escribe replay ni actualiza pesos.
Incluye la primera acción, los cambios del HUD y las transiciones terminales.

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

## Checkpoints y validación aislada

```sh
.venv/bin/python -m model train --checkpoint model/runs/square-cross/latest.pt --max-steps 1000000
.venv/bin/python -m model validate --checkpoint model/runs/square-cross/latest.pt
.venv/bin/python -m model validate --checkpoint model/runs/square-cross/latest.pt --levels 1 2 3 --validation-episodes 16
```

`latest.pt` se reemplaza atómicamente cada 1.000 acciones y al detenerse;
`best.pt` conserva la mejor evaluación visual del nivel actual. Guardan pesos,
ambos optimizadores, configuración, contadores, currículo y RNG. Al reanudar,
empieza una partida nueva con el replay vacío: recupera el entrenamiento, no la
misma trayectoria a mitad de partida. Los argumentos de entrenamiento dados
explícitamente sustituyen los guardados y reinician las rachas de aprobación.
`--output` permite ubicar checkpoints fuera del repositorio.
Los checkpoints anteriores al estado espacial no son compatibles con esta
arquitectura: se empieza un entrenamiento nuevo, sin adaptar pesos silenciosamente.

## Comprobaciones locales

```sh
.venv/bin/python -m unittest discover -s model/tests -v
```

Comprueban la aritmética/probabilidades del delta, el gradiente visual conjunto,
la equivalencia entre microbatches, la diversidad entre observaciones, la reanudación,
la validación congelada y las fronteras reales del SDK. No ejecutan navegador.
`architecture.json` conserva los IDs del diseño importado y describe el perfil
base. Declara cobertura parcial para los componentes futuros y no vincula los
pesos antiguos de LMEX a esta implementación.

Las pérdidas se inspiran en [SIGReg de LeJEPA](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md)
y en la [restricción de varianza de VICReg](https://github.com/facebookresearch/vicreg).
La combinación temporal, el muestreo de posiciones y sus pesos son decisiones
de esta implementación, no una reproducción de sus resultados publicados.
