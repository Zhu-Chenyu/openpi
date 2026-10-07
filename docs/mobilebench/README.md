# MobileBench π0.5

Design doc: [MobileBench_Method_v1_2_CurrentAffordance_DualMemory.md](MobileBench_Method_v1_2_CurrentAffordance_DualMemory.md)
(dual fast/slow latent memory, current/joint affordance, EEF goal + workspace decoders,
two action heads with bidirectional cross-attention, auxiliary losses).

Implementation (PyTorch π0.5 path):

- `src/openpi/models_pytorch/mobilebench/` — model (`model.py`), conditioner / memory
  (`conditioner.py`), workspace (`workspace.py`), dual action heads (`action_heads.py`),
  LoRA (`lora.py`), tests (`mobilebench_test.py`).
- `scripts/mobilebench/` — RMBench data prep (`prep_rmbench.py`), episode-sequence dataset
  (`rmbench_episodes.py`), trainer (`train_rmbench.py`), websocket eval server
  (`serve_rmbench.py`).
