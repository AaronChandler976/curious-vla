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
| `probe_cluster.sh` | Read-only listing of partitions, GPUs, quota, modules and internet access | piped over `ssh hpc3` |
| `submit_baseline.sh` | Submits 00, then 01 and the 02 sweep, which start only if 00 passed | login node |

## Which cluster

Use **HPC3** (`ssh hpc3`, node g17), which has A100s. HPC1 (`ssh hpc`, coe-hpc1.sjsu.edu) only has P100s, which current PyTorch no longer supports. The `hpc` and `hpc3` aliases are set up in `~/.ssh/config` on Aaron's Ubuntu desktop, with key auth and a ProxyJump through `hpc`. The SJSU VPN must be on.

## Steps

```bash
# 0. From the desktop, without cloning: a read-only look at HPC3's partitions, GPUs, quota and internet.
curl -fsSL https://raw.githubusercontent.com/AaronChandler976/curious-vla/hpc-scripts/hpc/probe_cluster.sh \
  | ssh hpc3 'bash -s' > hpc3_probe.txt
#    Set SLURM_PARTITION (and SLURM_GRES if needed) in hpc/config.sh from what it shows.

# 1. On HPC3:
ssh hpc3
git clone -b hpc-scripts https://github.com/AaronChandler976/curious-vla.git ~/curious-vla
cd ~/curious-vla
bash hpc/setup_env.sh            # once; run it on HPC3 itself, since home may not be shared with HPC1
bash hpc/submit_baseline.sh      # check job first, then the eval and fine-tunes if it passes
squeue -u $USER
```

Results go to `runs/` (git-ignored). Each eval writes `eval_info.json` with the success rate.

## Things that are not confirmed yet

The HPC web page is inconsistent and leaves these out. `probe_cluster.sh` gathers all of them in one go:

- **Partition on HPC3.** `SLURM_PARTITION` defaults to `gpu` and `SLURM_GRES` to `gpu:1`, which `submit_baseline.sh` passes to every job. Set them from `probe_cluster.sh` output. If you submit a single job by hand, add `-p <partition>`. `00_check` warns if it lands on a P100.
- **Time limits.** The page says 48 h maximum on GPU nodes and 4 h by default, so every job sets `--time`. Fine-tunes resume from their last checkpoint if you resubmit after a timeout.
- **Internet on compute nodes.** Not stated. `setup_env.sh` downloads everything on the login node. If jobs fail on downloads, add `export HPC_OFFLINE=1` to `config.sh`.
- **Headless rendering.** EGL is not mentioned. If `00_check` fails at the render step, set `MUJOCO_GL=osmesa` in `config.sh`. It is slower but needs no GPU display driver.
- **Camera key mapping.** `FINETUNE_RENAME_MAP` in `config.sh` maps the LIBERO camera names onto the ones SmolVLA was pretrained with. It has not been tested on a real run yet.
- **Home quota.** Not stated. The conda env plus the LIBERO dataset could be tens of GB. Run `du -sh ~/miniconda3 ~/.cache/huggingface` after setup.
