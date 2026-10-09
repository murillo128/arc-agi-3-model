# Política de entrenamiento — ARC-AGI-3

El proyecto utiliza **dos conjuntos locales** (entrenamiento y evaluación) y **dos fases de entrenamiento**. Kaggle proporciona la evaluación externa final.

## Fase 1: desarrollo local

- Seleccionar aleatoriamente **5 de los 25 juegos públicos** para evaluación completa, con semilla fija (`42`).
- En los **20 juegos restantes**, utilizar todos los niveles excepto el último para entrenamiento; reservar el último nivel de cada juego para evaluación.
- Entrenar y evaluar localmente tantas veces como sea necesario para elegir arquitectura, hiperparámetros y procedimiento de entrenamiento.
- Aplicar la misma separación a cualquier dato asociado (incluidas demostraciones humanas). No entrenar pesos con los juegos o niveles reservados durante esta fase.

## Fase 2: entrenamiento definitivo

- **Inicializar un modelo nuevo desde cero**, con pesos aleatorios. No cargar checkpoints ni hacer fine-tuning desde la fase 1.
- Reutilizar únicamente la arquitectura, configuración y procedimiento seleccionados.
- Entrenar con **los 25 juegos públicos y todos sus niveles**, incluidos los reservados antes para evaluación.
- Preparar el modelo resultante para Kaggle.

## Evaluación final

Enviar a **Kaggle** el modelo entrenado en la fase 2 y utilizar su evaluación privada como referencia independiente. No emplear datos privados de Kaggle para entrenamiento.

No habrá más particiones locales ni protocolos experimentales adicionales salvo que sean necesarios para desarrollar el agente.
