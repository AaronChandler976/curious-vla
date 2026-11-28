# Does the Source of Exploration Data Matter for Fine-Tuning a Small VLA?

**Team:** Aaron Chandler (CMPE 249 Intelligent Autonomous Systems, SJSU, Fall 2026)
**Track:** Research (VLA / robotics)
**Report:** [`report/main.tex`](report/main.tex) · **Slides:** [`slides/talk.pdf`](slides/talk.pdf) · **Video:** *link added with the final submission*

## Abstract

Vision-language-action (VLA) models learn manipulation from teleoperated demonstrations, which are expensive, while a robot can collect task-free exploration data almost for free. World-Env (arXiv:2509.24948) and TAP (ICML 2026, arXiv:2607.02466) show exploration data can help a VLA through reinforcement learning or inverse-dynamics pretraining. This project asks a narrower question: **under matched budgets, does the *source* of the exploration data matter when it is added to ordinary fine-tuning?** We fine-tune SmolVLA (450M parameters) on LIBERO-Object tasks 0 and 1 with 10, 25 or 40 demonstrations per task and compare:

1. **Demos only.**
2. **Demos + random exploration** (uniform actions held for 5 steps).
3. **Demos + ICM exploration** (PPO on the Intrinsic Curiosity Module reward, Pathak et al. 2017).

Exploration is matched by collected environment steps (5k and 20k), training by gradient steps (20k), and every cell has 10 training seeds.

## Results so far

239 fine-tunes, each scored on 100 rollouts, replanning every 10 actions (`n_action_steps=10`; see below). Full tables: `runs/analysis_nas10/summary.md`, made by `hpc/analyze_results.py --eval-dir eval_nas10`.

- **Exploration data never helps when co-trained with demos.** The paired change vs demos only, pooled over demo budgets, is −11.9 (random 5k), −9.3 (random 20k), −9.6 (ICM 5k) and −11.5 (ICM 20k) points. It hurts most with 10 demos per task (−10 to −25).
- **The exploration source makes no measurable difference.** ICM minus random is +2.3 [−3.2, +7.7] at 5k and −2.3 [−8.0, +3.4] at 20k, and re-collecting the same method's data moves results as much as switching methods.
- **The evaluation setting mattered more than any method.** Executing each predicted 50-action chunk open loop (the checkpoint default) scored 18 to 38 points lower and doubled the seed noise. The official `smolvla_libero` checkpoint goes from 64% to 82% on these tasks. Under that noisier evaluation, ICM vs random looked like +12 / −13 points, which the closed-loop re-scoring shows was noise.
- **Why:** SmolVLA fine-tunes only its action expert (the VLM is frozen), and exploration actions are not predictable from observations (held-out R² ≤ 0.01 vs 0.42 for demos; `hpc/exploration_stats.py`). Early falsification runs show that halving the demo gradient steps costs nothing, while the mixes stay below that control, so exploration frames actively hurt rather than just dilute.
- **Still running:** two-stage (exploration first, then demos), an ICM explorer with a 150k-step warm-up, and 3 independent collections per method. [`docs/research-review.md`](docs/research-review.md) sets out each test and what each outcome would mean.

| Condition (10 seeds each) | 10 demos | 25 demos | 40 demos |
|---|---:|---:|---:|
| Demos only | 78.2 | 83.3 | 90.1 |
| + random 5k | 68.6 | 63.7 | 83.6 |
| + random 20k | 57.3 | 82.6 | 83.9 |
| + ICM 5k | 56.0 | 79.1 | 87.6 |
| + ICM 20k | 53.4 | 79.5 | 84.1 |

Task success (%), replanning every 10 actions. The official `HuggingFaceVLA/smolvla_libero` checkpoint scores 82% on the same two tasks.

## Repository structure

| Path | Contents |
|---|---|
| `hpc/` | The whole experiment pipeline for the SJSU CoE HPC (Slurm, offline nodes); see [`hpc/README.md`](hpc/README.md) |
| `hpc/plan.txt` | Every run in the study, one per line; `hpc/autopilot.sh` keeps it running |
| `hpc/collect_exploration.py` | Random, ICM and warmed-up ICM explorers, recording in the demo dataset's exact format |
| `hpc/build_dataset.py`, `hpc/verify_training_data.py` | Build each training set from raw rows (demos only, mix, or exploration only) and verify every sampled frame |
| `hpc/02_finetune_demos.sbatch` | One fine-tune (co-training, two-stage, or demos only) followed by the LIBERO evaluation |
| `hpc/analyze_results.py` | Per-cell means, hierarchical-bootstrap CIs, paired comparisons, vector figures |
| `hpc/exploration_stats.py`, `hpc/train_profile.py`, `hpc/profile_inference.py` | Exploration-data diagnostics, training cost, inference latency and memory |
| `docs/research-review.md` | Novelty and state-of-the-art check, the proposal feedback, falsification plan |
| `literature-survey.md`, `ai-novelty-feasibility-audit.md` | Proposal-stage survey and AI novelty audit |
| `slides/` | Talk (Beamer); `hpc/talk_figures.py` makes its figures |
| `report/` | Final report (LaTeX, IEEE conference format) |
| `runs/` | Logs, checkpoints and results; git-ignored, created on the cluster |

## Environment setup

Pinned: `lerobot[smolvla,libero]==0.7.0`, Python 3.12, torch 2.9.1 (CUDA 12.6 build). HPC3 has no internet, so the environment, models, demos and LIBERO assets are built on a machine that has it and copied over:

```bash
git clone -b hpc-scripts https://github.com/AaronChandler976/curious-vla.git && cd curious-vla
bash hpc/stage_from_desktop.sh                   # packed env, HF cache, LIBERO assets -> HPC3
ssh hpc3 'cd ~/curious-vla && bash hpc/verify_cache.sh && sbatch hpc/dry_run.sbatch'
```

On a machine with internet and a GPU, `bash hpc/setup_env.sh` installs the same environment in place, and every script runs without Slurm.

## Training and evaluation

```bash
# Everything in hpc/plan.txt, unattended (cron every 20 min on the login node):
bash hpc/autopilot.sh
# One run by hand, e.g. 10 demos/task + 5k random exploration steps, seed 0:
EXPLORE_METHOD=random EXPLORE_STEPS=5000 N_DEMOS=10 SEED=0 sbatch hpc/02_finetune_demos.sbatch
# Collect exploration data (random | icm | icmw):
EXPLORE_METHOD=icm sbatch hpc/03_collect.sbatch
# Results and figures:
python hpc/analyze_results.py runs              # -> runs/analysis/
```

## Inference

Any checkpoint can be run in the LIBERO simulator:

```bash
lerobot-eval --policy.path=runs/<run>/checkpoints/last/pretrained_model \
  --env.type=libero --env.task=libero_object --env.task_ids='[0,1]' --eval.n_episodes=50
python hpc/profile_inference.py --policy runs/<run>/checkpoints/last/pretrained_model   # latency and memory
```

## Models and data

- Base model: [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base), given LIBERO's feature format by `hpc/make_base_policy.py`
- Reference checkpoint: [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
- Demonstrations: [`HuggingFaceVLA/libero`](https://huggingface.co/datasets/HuggingFaceVLA/libero); simulator assets: [`lerobot/libero-assets`](https://huggingface.co/datasets/lerobot/libero-assets)
- Our fine-tuned checkpoints and exploration datasets live on the SJSU HPC (`runs/`); links will be added here with the final submission.

## Compute

SJSU CoE HPC: A100 40GB and H100 nodes. One 20k-step fine-tune takes 1.0 h on an H100 (1.6 h on an A100) at 7.7 GB peak memory; 20k exploration steps take about 32 minutes.
