# Comparing Exploration Sources for Sample-Efficient VLA Fine-Tuning

**Team:** Aaron Chandler — CMPE 249, Intelligent Autonomous Systems, Fall 2026
**Selected track:** Research (VLA / robotics lane)

## Abstract

Vision-Language-Action (VLA) models for robot manipulation are trained almost entirely on task-labeled teleoperated demonstrations. Those demonstrations are expensive, and interaction data without a task label is usually thrown away. World-Env (arXiv:2509.24948) has already shown that autonomously collected exploration data can improve VLA fine-tuning on LIBERO. This project asks a narrower question: **does the source of that exploration data matter?** Under identical budgets, we compare three conditions for fine-tuning SmolVLA on LIBERO-Object tasks:

1. **Demos only.** A reduced budget of task demonstrations (10, 25 or 50 per task).
2. **Demos + random exploration.** Undirected random play in the same scenes.
3. **Demos + ICM exploration.** Curiosity-driven play from a policy trained on the Intrinsic Curiosity Module reward (Pathak et al., 2017).

Random and ICM exploration are matched by environment steps (5k and 20k). All conditions use the same base model, training steps, normalization statistics and evaluation. We report task success against the number of demonstrations and the amount of exploration data. Everything runs in simulation (LIBERO + SmolVLA via LeRobot). VLM-guided curiosity is a stretch goal.

## Repository structure

- `literature-survey.md` — SOTA survey (10 papers and models, 2023–2026)
- `ai-novelty-feasibility-audit.md` — AI-assisted novelty and red-ocean critique of the topic
- `hpc/` — the full experiment pipeline for the SJSU CoE HPC (Slurm) cluster; see [`hpc/README.md`](hpc/README.md)
  - environment staging for an offline cluster, a GPU check and a CPU dry run of every code path
  - the reference eval of `HuggingFaceVLA/smolvla_libero`, demos-only fine-tunes, random and ICM exploration collection, and mixed fine-tunes
  - `status.sh` and `summarize_results.py` for progress and the results table
- `runs/` — logs, checkpoints and results; git-ignored, created on the cluster

## Environment setup

The pipeline pins `lerobot[smolvla,libero]==0.7.0`, Python 3.12 and torch 2.9.1 built for CUDA 12.6. The cluster has no internet access, so the environment and all data are built on a machine that has it and copied over:

```bash
git clone -b hpc-scripts https://github.com/AaronChandler976/curious-vla.git ~/curious-vla
cd ~/curious-vla && bash hpc/stage_from_desktop.sh      # env, models, LIBERO demos and assets → HPC3
ssh hpc3 'cd ~/curious-vla && bash hpc/verify_cache.sh && bash hpc/dry_run.sh'
```

On a machine with internet, `bash hpc/setup_env.sh` installs the same environment in place.

## Running training and evaluation

```bash
ssh hpc3
cd ~/curious-vla
bash hpc/submit_baseline.sh       # reference eval + demos-only fine-tunes (each job trains, then evaluates)
bash hpc/submit_exploration.sh    # random/ICM collection + demos+exploration fine-tunes
bash hpc/status.sh                # queue, errors and the results table (runs/results.csv)
```

To evaluate any checkpoint (inference in the LIBERO simulator):

```bash
lerobot-eval --policy.path=runs/<run>/checkpoints/last/pretrained_model \
  --env.type=libero --env.task=libero_object --env.task_ids='[0,1]' --eval.n_episodes=50
```

## Models and datasets

- Base model: [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base), started with LIBERO's feature format (`hpc/make_base_policy.py`)
- Reference checkpoint: [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
- Demonstrations: [`HuggingFaceVLA/libero`](https://huggingface.co/datasets/HuggingFaceVLA/libero)
- Simulator assets: [`lerobot/libero-assets`](https://huggingface.co/datasets/lerobot/libero-assets)
- Our fine-tuned checkpoints and exploration datasets: to be linked here after the runs finish

## Status (Nov 23, 2026)

- The pipeline runs end to end on HPC3's A100 nodes.
- The official `smolvla_libero` checkpoint scores 64% on LIBERO-Object tasks 0–1 (100 episodes) through this pipeline.
- Demos-only fine-tunes at 10, 25 and 50 demos per task are running.
- Random and ICM exploration are being dry-run before collection.
