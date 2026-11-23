# Running on the SJSU CoE HPC

These scripts run the first slice of the project on the CoE HPC cluster, which uses the Slurm scheduler:

1. A GPU node check.
2. An evaluation of the official SmolVLA LIBERO checkpoint, as a sanity check of the pipeline.
3. The **demos-only** fine-tunes at reduced demo budgets.

The random and ICM exploration conditions will add their own jobs later and reuse the same config.

| File | What it does | Where it runs |
|---|---|---|
| `config.sh` | All settings: paths, versions, suite and task ids, rendering | sourced by everything |
| `stage_from_desktop.sh` | Builds the env on the desktop, packs it with conda-pack, downloads models, demos and LIBERO assets, copies it all to HPC3 and unpacks it | **desktop**, once |
| `setup_env.sh` | The same setup done in place, for a machine that has internet (not HPC3) | machine with internet |
| `00_check.sbatch` | GPU, PyTorch-on-GPU and headless LIBERO render check (~10 min) | GPU node |
| `01_eval_reference.sbatch` | `lerobot-eval` of `HuggingFaceVLA/smolvla_libero` on our tasks | GPU node |
| `02_finetune_demos.sbatch` | Fine-tunes `lerobot/smolvla_base` on N demos per task, then evaluates it; resumes after a timeout | GPU node |
| `make_base_policy.py` | Copies `smolvla_base` with the official LIBERO checkpoint's feature names and shapes, checked against the dataset | called by 02 and the dry run |
| `select_episodes.py` | Picks N demo episodes per task (seeded) and records the choice in `runs/selections/` | called by 02 |
| `probe_cluster.sh` | Read-only listing of partitions, GPUs, quota, modules and internet access | piped over `ssh hpc3` |
| `verify_cache.sh` | Offline check that every model, the dataset and the LIBERO assets are cached | HPC3 login node |
| `dry_run.sh` | Runs every job's code path tiny and on CPU, offline, in ~10 min: catches setup errors before queueing | HPC3 login node |
| `03_collect.sbatch` | Collects random or ICM exploration data (rendering only needs a GPU), then runs the format check | GPU node, any type |
| `collect_exploration.py` | The random and ICM (PPO on the ICM curiosity reward) explorers, recording in the demo dataset's exact format | called by 03 |
| `check_exploration_format.py` | Compares exploration data with the demos: features, first-frame state, image orientation, action range | called by 03 and the dry run |
| `build_mix.py` | Selected demos + the first N exploration steps, demos repeated to ~50% of frames, demo normalization stats | called by 02 |
| `submit_exploration.sh` | Submits both collections and the mixed fine-tunes that depend on them | HPC3 login node |
| `status.sh` / `summarize_results.py` | Queue, recent job states, last error per log, and the results table (`runs/results.csv`) | HPC3 login node |
| `cleanup_run.sh` | Keeps only the final and 10k-step weights of a finished run | called by 02 |
| `submit_baseline.sh` | Submits 00, then 01 and the 02 sweep, which start only if 00 passed | login node |

## Which cluster

Submit from **HPC3** (`ssh hpc3`, login node g17). The `hpc` and `hpc3` aliases are in `~/.ssh/config` on Aaron's Ubuntu desktop (key auth, ProxyJump through `hpc`), and the SJSU VPN must be on. A probe run on Nov 22, 2026 found:

- **GPU partitions:** `gpuqs` (max 2 days), `gpuqm` (7 days) and `gpuql` (14 days). All three cover the same nodes: 4 A100 nodes (cs001-cs004), 3 H100 nodes (g16, g18, g19) and P100 nodes, which are too old for current PyTorch. The jobs request `gpuqs` with `--gres=gpu:a100:1`. Naming the type keeps you off the P100s. Change it in `config.sh`. If the A100 queue is long, use `SLURM_GRES=gpu:h100:1`, or try the NSF H100 queue with `SLURM_PARTITION=nsfq SLURM_GRES=gpu:h100:1 bash hpc/submit_baseline.sh` (13 H100 nodes, g20-g32, max 24 h per job and 6 nodes per user).
- **No internet** on g17, so `git clone`, `pip` and Hugging Face downloads fail there. Everything is staged from the desktop, and jobs run with `HF_HUB_OFFLINE=1`.
- **No conda or CUDA modules,** but Apptainer is installed. The env is a conda-pack tarball that brings its own Python and CUDA 12.6 runtime. The driver is 560.35 (CUDA 12.6), which is why torch is the cu126 build.
- **/scratch is wiped 21 days into each semester,** so everything lives in home instead.
- **Shared /home is 96% full** (5.4 TB free cluster-wide, no per-user quota shown). Delete old checkpoints in `runs/` when you no longer need them.

## Steps

```bash
# 1. On the desktop, VPN on (network and CPU only, so it can run while the GPU is busy):
git clone -b hpc-scripts https://github.com/AaronChandler976/curious-vla.git ~/curious-vla
#   (already have ~/curious-vla?  cd ~/curious-vla && git fetch origin && git switch hpc-scripts && git pull)
cd ~/curious-vla && bash hpc/stage_from_desktop.sh

# 2. On HPC3:
ssh hpc3
cd ~/curious-vla && bash hpc/submit_baseline.sh
squeue -u $USER
# Once the check job has passed on HPC3, later submissions can skip it:
#   SKIP_CHECK=1 bash hpc/submit_baseline.sh
```

To update the scripts on HPC3 later, run `git pull` on the desktop and then re-run `stage_from_desktop.sh`. rsync only sends the changed files.

Results go to `runs/` (git-ignored). Each eval writes `eval_info.json` with the success rate. To bring results back to the desktop, run `rsync -a hpc3:curious-vla/runs/ ~/curious-vla/runs/`.

## Still unconfirmed

- **torch 2.9.1+cu126 on the compute nodes.** The A100 node's driver may differ from g17's. `00_check` prints both, and every other job waits for it to pass.
- **Headless rendering.** If `00_check` fails at the render step, set `MUJOCO_GL=osmesa` in `config.sh`. It is slower, but it needs no GPU display driver.
- **Feature format.** Fine-tunes start from a copy of `smolvla_base` whose config has the official LIBERO checkpoint's features: cameras `image`/`image2`, an 8-dim state and a 7-dim action. That copy is made by `make_base_policy.py`, which also checks the features against the dataset. Without it, training silently kept smolvla_base's 6-dim state, which the dry run caught. `dry_run.sh` checks that the trained checkpoint's features match the official checkpoint's.
- **Offline dataset loading.** If a job fails on a Hugging Face download, the cache copy is incomplete. Re-run `stage_from_desktop.sh`.
