# Training Policy — ARC-AGI-3

The project uses **two local datasets** (training and evaluation) and **two training phases**. Kaggle provides the final independent evaluation.

## Phase 1: Local development

- Randomly select **5 of the 25 public games** for full-game evaluation, using a fixed seed (`42`).
- For the **remaining 20 games**, use every level except the final one for training; reserve each game's final level for evaluation.
- Train and evaluate locally as often as needed to select the architecture, hyperparameters, and training procedure.
- Apply the same split to all associated data (including human demonstrations). Do not train model weights on held-out games or levels during this phase.

## Phase 2: Final training

- **Initialize a new model from scratch** with random weights. Do not load Phase 1 checkpoints or fine-tune the Phase 1 model.
- Reuse only the selected architecture, configuration, and training procedure.
- Train on **all 25 public games and all their levels**, including those previously reserved for evaluation.
- Prepare the resulting model for Kaggle.

## Final evaluation

Submit the Phase 2 model to **Kaggle** and use its private evaluation as the independent final benchmark. Do not use Kaggle's private data for training.

No additional local splits or experimental protocols are required unless necessary to develop the agent.
