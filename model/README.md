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

El modelo **base** tiene 214.139.520 parámetros:

| Componente | Configuración | Parámetros |
|---|---|---:|
| Visual encoder | 12 bloques, ancho 768, 12 cabezas, patches 4×4; memoria de animación 1024 | 91.521.792 |
| Dynamics | Memoria 2048; predictor de 6 bloques, ancho 768; latente 1024 | 64.929.152 |
| Visual decoder | 8 bloques, ancho 768, 12 cabezas; 16 colores | 57.688.576 |

`--profile small --device cpu --no-compile` usa la misma computación con
74.776 parámetros para desarrollo. No es un sustituto de capacidad equivalente.
`--frame-chunk` limita frames por llamada espacial; `--batch-episodes` controla
cuántas partidas se muestrean como mínimo. El perfil base no se reduce
automáticamente al encontrar límites de memoria.

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
4. Reconstruye la memoria causal `h` desde el inicio de cada partida. El predictor
   recibe `z_t, h_t, a_t` y calcula `h_(t+1), d_hat_(t+1)`;
   **`z_hat_(t+1) = z_t + d_hat_(t+1)`**, sin normalización posterior a la suma.
5. Entrena encoder y dynamics con `.98 * MSE(d_hat, z_(t+1)-z_t) + .02 * SIGReg`.
   Ambos latentes reales conservan gradiente. SIGReg usa al menos 32
   observaciones reales distintas, proyecciones aleatorias y cálculo FP32.
   No hay profesor EMA ni etiquetas tomadas de trayectorias imaginadas.
6. El decoder recibe `z_hat.detach()`, produce logits de 16 colores por píxel y
   aprende del tablero final real. Su gradiente no entra al predictor/encoder.
   La CE equilibra celdas cambiadas y no cambiadas; si ninguna cambia usa CE
   ordinaria. La salida visual delta compara el tablero decodificado con el
   tablero anterior: `KEEP=16`, o nuevo color 0…15. Los píxeles del tablero
   anterior **no entran a la red del decoder**; solo definen geometría y comparación.

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
- MSE del delta latente ≤ .01 y desviación típica media del latente ≥ .1, para
  impedir que un encoder colapsado apruebe por producir siempre el mismo vector.
- Ninguna transición omitida por falta de frames o incompatibilidad geométrica.

Son valores iniciales configurables, no resultados de convergencia demostrados.
Los eventos JSON de terminal muestran acciones, actualizaciones, pérdidas,
errores, partidas, acciones/s, memoria GPU máxima y checkpoints. Victorias y
GAME_OVER se reportan aparte de la calidad predictiva.

Se detiene guardando `latest.pt` al llegar a 100.000 acciones sin converger en
un nivel, a 500.000 acciones totales, al completar el currículo o con Ctrl+C.
Los límites se ajustan mediante `--max-level-steps` y `--max-steps`.

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

## Comprobaciones locales

```sh
.venv/bin/python -m unittest discover -s model/tests -v
```

Comprueban la aritmética del delta, los gradientes separados, la reanudación,
la validación congelada y las fronteras reales del SDK. No ejecutan navegador.
`architecture.json` conserva los IDs del diseño importado y describe el perfil
base. Declara cobertura parcial para los componentes futuros y no vincula los
pesos antiguos de LMEX a esta implementación.
