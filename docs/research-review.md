# Research review (Nov 27, 2026)

A full re-check of the project after the main grid finished (224 runs), done the way the class-wide
proposal feedback asks: check the real state of the art, find the strongest baseline, design
falsification tests, list the confounds, and decide in advance what each outcome would mean.
Each question below has a **criterion** that settles it and a **status**. New runs launched on
Nov 27 are in `hpc/plan.txt` (lines after "Added 2026-11-27").

## 1. Novelty, scope and state of the art

Searches on Nov 27, 2026 (arXiv, project pages, code repositories) for exploration, curiosity,
play and task-agnostic data used to train or fine-tune VLAs.

| Work | What it shows | How this project differs |
|---|---|---|
| **TAP**, Shi et al., ICML 2026 (arXiv 2607.02466, code released) | Task-agnostic pretraining on off-task trajectories and ~30 h of autonomous random play with an **inverse-dynamics** loss, then BC on a small expert set: +10 pts Avg-All on SIMPLER (23.2 to 33.3), gains grow with more play data. Not on LIBERO; one exploration strategy only; no co-training comparison. | Closest new work. It settles "random play can help a VLA" when it enters through inverse dynamics. It does not compare exploration sources, and it does not test plain BC co-training or sequential BC on the play actions, which is how a LeRobot user would add such data. |
| **World-Env**, Xiao et al. 2025/26 (arXiv 2509.24948) | Exploration from a perturbed, already-trained OpenVLA-OFT policy, used for RL post-training inside a learned world model; LIBERO, 5 demos per task. | Exploration source is a competent policy plus noise, and the channel is RL with a VLM reward. We compare task-free sources (random vs curiosity) through the BC channel. |
| **ExORL**, Yarats et al. 2022 | For offline RL with reward relabeling, the exploration method that collects the data (ICM, RND, APT, random, ...) changes downstream performance a lot. | The closest "does the source matter" study, but with reward labels and offline RL. VLA fine-tuning has no reward; this project asks the same question for imitation. |
| **DLR**, 2025 (arXiv 2511.19528) | Diverse RL-generated trajectories for VLA pretraining on LIBERO. | Multi-pattern RL with task reward, not task-free exploration. |
| PDE, ExToken (2026) | Exploration inside VLA RL fine-tuning. | Exploration as an RL algorithm, not as data. |
| Negative transfer and failure-data papers (2026, e.g. arXiv 2607.01586, FACT 2608.10232) | Pooling mismatched data under action-only supervision can make a VLA worse than no pretraining; failed rollouts help when used as negatives or recovery data, not as BC targets. | Predicts our main negative result. |

**Verdict.** No paper found compares curiosity-driven vs random exploration data as VLA
fine-tuning data under matched budgets, so the narrowed question is still open. It is not a red
ocean, but it is narrower than the original proposal: "exploration data can help" is shown (World-Env
via RL, TAP via inverse dynamics). The defensible contribution is (a) the source comparison at
matched collected steps, (b) which training channel (co-training vs exploration-first) can carry
any value under the default SmolVLA recipe, and (c) a well-diagnosed negative result.

**State of the art in the toolchain.** SmolVLA (2025, 450M params, flow-matching action expert) via
LeRobot 0.7.0 on LIBERO is current and standard. ICM (2017) is old on purpose: the professor asked
for "inexpensive" exploration. Stronger intrinsic rewards (RND, disagreement, APT) and TAP's inverse
dynamics loss are not run and are listed as future work. Ten training seeds with hierarchical
bootstrap intervals is more statistical care than most VLA papers, which report one seed.

**Reference check.** The official `smolvla_libero` checkpoint scores 64% on our two tasks, below the
paper's ~96% for LIBERO-Object. Other groups see the same gap: one paper reports 65.6% for SmolVLA
on LIBERO-Object, and a reproduction study notes that the public checkpoint was trained on one
eighth of the paper's sample budget. So the gap comes from the public checkpoint, not our pipeline. The
`n_action_steps` ablation (run `05_profile_and_variants`) checks how much open-loop chunk
execution costs.

**Scope.** It fits a semester and the available compute: one suite, two tasks, a 450M model,
about 2 GPU-hours per run on A100/H100. 464 runs is affordable because the autopilot keeps up to
about 8 GPUs busy.

## 2. The professor's feedback on this proposal

| Ask | Status |
|---|---|
| Reproduce SmolVLA fine-tuning on 1-2 LIBERO tasks with a reduced demo budget | Done: LIBERO-Object tasks 0-1, 10/25/40 demos per task, reference eval of the official checkpoint through the same pipeline. |
| Demos only vs + random vs + ICM under exactly the same exploration-data and training budgets | Done for the 5k and 20k env-step budgets, same 20k training steps, batch 32, base model, normalization stats, eval seeds. Caveat found in review: the ICM explorer made only 7 PPO updates (policy std stayed 1.0), so "ICM" was close to a second random explorer with a different action distribution. Fixed by `icmw` (150k unrecorded warm-up steps, about 230 updates). Its recorded budget is the same, and the extra interaction is reported. |
| Plot success vs number of demos and vs amount of exploration data | Done: `fig_success_vs_demos.pdf`, `fig_success_vs_exploration.pdf`. |
| VLM-guided curiosity only as a stretch goal | Not started, on purpose. Co-training with any exploration data loses success, and the BC action loss is the only path for that data (the VLM is frozen). Better-labeled curiosity data sent down the same path would not answer anything. The two-stage runs test whether another path can carry value first. |
| Key question: do inexpensive strategies differ in downstream value per unit of collected data? | Answered per unit of data: the value is negative for both sources when co-trained (Section 4). The ICM-vs-random difference flips sign between 5k and 20k and is within the spread between exploration collections, so the replicate runs test it with the collection as the unit. |

## 3. The class-wide questions, with criteria

**Q1. Who has already done the closest thing?**
Criterion: for each part of the claim (exploration source, VLA, LIBERO, BC channel), find the
nearest paper and read its full text, not only the abstract. *Status: done (table above). TAP and
World-Env full text checked; TAP was added on Nov 27.*

**Q2. What is the strongest baseline that could make the method unnecessary?**
- *More demos.* The demo-scaling curve gives the exchange rate. Criterion: exploration is
  worth collecting only if a mix beats demos only at the same demo budget, by more than the gain
  from adding demos. Measured: demos only 46 / 45 / 58% at 10 / 25 / 40 demos.
- *Random exploration* is the baseline that makes curiosity unnecessary. Criterion: ICM is
  justified only if ICM > random with a 95% CI above 0, pooled over demo budgets, **and** in the
  same direction on every exploration collection. Status: fails at 20k (−13 pts) and passes at 5k
  (+12 pts) on one collection. The replicate runs settle it.
- *Demos only with the same compute.* Every condition has 20k training steps. A 50% mix gives
  the demos half the gradient steps, so the matched-exposure control (demos only, 10k steps) is
  now in the plan.
- *TAP-style inverse dynamics* and *World-Env-style perturbed-policy exploration* would be stronger.
  They are out of scope for the remaining time and are named as future work. The two-stage
  condition is a cheap step toward TAP: exploration first, demos second.

**Q3. What experiment could falsify the hypothesis?**
H1 is "exploration data adds value to SmolVLA fine-tuning". Planned falsifiers and their readings:
1. *Dilution control* (`demos N S 10000`): demos only for 10k steps, paired with the mixes by seed.
   - mix ≈ demos-10k (paired CI within ±5 pts): exploration frames are inert, and the whole loss is dilution.
   - mix < demos-10k: exploration frames actively hurt (negative transfer).
   - mix > demos-10k: they carry some value that dilution hides.
2. *Dose-response* (mixing-ratio ablation, done): if harm grows with the exploration share, that
   supports dilution or harm. Seen: at 10% exploration, results return to the demos-only level.
3. *Action predictability* (`exploration_stats.py`): if the actions cannot be predicted from the
   observation (held-out R² ≤ 0), the BC loss cannot extract an observation-to-action mapping
   from them. Measured: R² ≤ 0.01 for every random and ICM collection, against +0.42 for the demos. Only 11 to 18% of exploration frames come within 3 cm of anywhere the demos go.
4. *Exploration-first* (`seq`): if exploration data has any value through BC, a 10k-step stage
   on it before 10k demo steps should beat demos-10k (the same demo exposure).
H2 is "curiosity data is worth more than random data". Falsifier: `icmw` vs random at both budgets on
three collections each. If a properly trained ICM is no better than random, H2 is rejected for
this channel.

**Q4. What confound would make the current comparison invalid?**
| Confound | Effect | Control |
|---|---|---|
| Halved demo exposure in a 50% mix | Mix looks worse even if exploration is inert | demos-10k control, ratio ablation |
| ICM barely trained (7 updates) | "ICM vs random" measured action distributions, not curiosity: clipped Gaussian, mean abs(a) 0.63 vs uniform 0.50, about 25% more 2 cm workspace voxels visited at 20k | `icmw`; exploration stats per dataset |
| One exploration collection per method in the main grid | Training seeds are not replicates of the data source (pseudo-replication). Replicates already differ by up to 25 pts (random 5k n10: 36 / 35 / 11%) | 3 collections per method, analysis with the collection as unit |
| The 5k data is the first 5k steps of the 20k collection | Budgets are nested, not independent | Stated; replicates give independent draws |
| The seed sets demo subset, training noise and eval initial states together | Demo subset could drive the variance | Paired analysis by seed (pairing barely shrinks the CIs, so training noise dominates the subset) |
| 100 eval episodes per run | ±5 pts binomial SE per run | Hierarchical bootstrap over seeds and episodes |
| LR schedule auto-scales to total steps | 10k-step runs decay faster | Matched where it matters: demos-10k and seq stage 2 have the same schedule |
| Frozen VLM (`train_expert_only`) | Exploration cannot change visual features, only the action expert and state projection | Scope statement; unfreezing is future work |
| Open-loop chunks (n_action_steps 50) | May depress all success rates | Same for all conditions; ablation at 10 and 25 |

**Q5. What is the cheapest experiment to run this week to decide whether to continue?**
`exploration_stats.py` (one CPU minute) showed the exploration actions carry no
observation-dependent signal (R² ≤ 0.01 vs 0.42 for demos) and the untrained ICM differs from random mainly in action scale. The
demos-10k control costs half a run. Both ran first.

**Q6. If the method does not improve the metric, is the result still meaningful?**
Yes, if the null is tight and explained. Criteria: (a) report paired 95% CIs and call a
difference "practically zero" only if the CI lies within ±5 pts (about one eval SE); (b) name the
mechanism with evidence (dilution vs active harm, from the demos-10k control; no learnable signal,
from R²); (c) state what would change it (an inverse-dynamics channel, a trained explorer, unfreezing
the VLM). The current result already says something practical: adding task-free exploration data to
a SmolVLA fine-tune the default way costs up to 19 pts, so don't do it.

## 4. Results so far (224 runs, Nov 26)

- No exploration mix beats demos only at any demo budget. Pooled differences: random 5k −13.7,
  random 20k −5.7, ICM 5k −1.7, ICM 20k −18.9 pts.
- ICM vs random: +12.0 pts at 5k, −13.2 pts at 20k (pooled). The sign flips with budget, and the
  spread between collections is as large, so neither supports a data-source effect yet.
- At 10% exploration frames the mix is back at the demos-only level (dilution-consistent).
- Paired tables: `runs/analysis/summary.md` (section "Paired comparisons").

## 5. What was added on Nov 27 (all on HPC3)

| Run family | Runs | Question |
|---|---:|---|
| `demos N S 10000` | 30 | Dilution vs active harm |
| `seq random/icm 20000` | 60 | Exploration first, then demos (TAP-like order, BC loss) |
| `mix icmw 5000/20000`, `seq icmw` | 90 | Does a trained curiosity explorer change anything? |
| Collection replicates (random/icm/icmw, e1-e3) | 60 | ICM vs random with the data draw as unit |
| `05_profile_and_variants` | 1 job | Latency and memory; n_action_steps 10/25 |
