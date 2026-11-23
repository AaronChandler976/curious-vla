# Running on the SJSU CoE HPC

These scripts run the first slice of the project on the CoE HPC cluster, which uses the Slurm scheduler:

1. A GPU node check.
2. An evaluation of the official SmolVLA LIBERO checkpoint, as a sanity check of the pipeline.
3. The **demos-only** fine-tunes at reduced demo budgets.

The random and ICM exploration conditions will add their own jobs later and reuse the same config.

| File | What it does | Where it runs |
|---|---|---|
| `config.sh` | All settings: paths, versions, suite and task ids, rendering | sourced by everything |
| `setup_env.sh` | Installs Miniconda, the env and LeRobot 0.7.0; downloads models, LIBERO assets and the demo dataset | **login node**, once |
| `00_check.sbatch` | GPU, PyTorch-on-GPU and headless LIBERO render check (~10 min) | GPU node |
| `01_eval_reference.sbatch` | `lerobot-eval` of `HuggingFaceVLA/smolvla_libero` on our tasks | GPU node |
| `02_finetune_demos.sbatch` | Fine-tunes `lerobot/smolvla_base` on N demos per task, then evaluates it; resumes after a timeout | GPU node |
| `select_episodes.py` | Picks N demo episodes per task (seeded) and records the choice in `runs/selections/` | called by 02 |
| `submit_baseline.sh` | Submits 00, then 01 and the 02 sweep, which start only if 00 passed | login node |

## Steps

```bash
# On campus or on the SJSU VPN:
ssh <SJSU_ID>@coe-hpc1.sjsu.edu
git clone https://github.com/AaronChandler976/curious-vla.git ~/curious-vla
cd ~/curious-vla

bash hpc/setup_env.sh            # once, on the login node; downloads a lot, so check your quota
sbatch hpc/00_check.sbatch       # wait for it, then read runs/logs/cvla-check-<id>.out
bash hpc/submit_baseline.sh      # or skip the separate check: this script runs it first anyway
squeue -u $USER
```

Results go to `runs/` (git-ignored). Each eval writes `eval_info.json` with the success rate.

## Things that are not confirmed yet

The HPC web page is inconsistent and leaves these out. Check them on the cluster with `sinfo`, `scontrol show partition gpu` and `module avail`:

- **Partition and GPU type.** The scripts use `--partition=gpu --gres=gpu:1`. That partition mixes P100, A100 and H100 nodes. A P100 is too old for current PyTorch builds, and `00_check` warns if you get one. If `sinfo` shows a way to ask for a specific type (for example `--gres=gpu:a100:1` or a per-model partition), set it in the `#SBATCH` lines.
- **Time limits.** The page says 48 h maximum on GPU nodes and 4 h by default, so every job sets `--time`. Fine-tunes resume from their last checkpoint if you resubmit after a timeout.
- **Internet on compute nodes.** Not stated. `setup_env.sh` downloads everything on the login node. If jobs fail on downloads, add `export HPC_OFFLINE=1` to `config.sh`.
- **Headless rendering.** EGL is not mentioned. If `00_check` fails at the render step, set `MUJOCO_GL=osmesa` in `config.sh`. It is slower but needs no GPU display driver.
- **Camera key mapping.** `FINETUNE_RENAME_MAP` in `config.sh` maps the LIBERO camera names onto the ones SmolVLA was pretrained with. It has not been tested on a real run yet.
- **Home quota.** Not stated. The conda env plus the LIBERO dataset could be tens of GB. Run `du -sh ~/miniconda3 ~/.cache/huggingface` after setup.
