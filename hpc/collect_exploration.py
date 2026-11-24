"""Collect task-free exploration data in LIBERO, in exactly the demo dataset's format.

Two exploration sources, run under the same environment-step budget:
  random : undirected random play. A uniform random action in [-1, 1]^6 plus a random gripper
           command (+1 or -1), each held for --hold steps so the arm actually travels.
  icm    : curiosity-driven play. A small Gaussian policy is trained online with PPO on the
           Intrinsic Curiosity Module reward (Pathak et al. 2017): the forward model's error at
           predicting the next state's learned features. It acts with the same --hold.

Both read the same low-dimensional observation (the 8-D robot state plus the simulator's joint
positions, which include every object's pose). Neither sees images or task rewards.

Every env step is recorded with the observation the policy would see in evaluation: LeRobot's
own LIBERO wrapper and its canonical_observation(), which rotate the images by 180 degrees and
build the 8-D state the same way the demos were converted. Features, fps and dtypes are copied
from the demo dataset's info.json, so the result can be merged with the demos. All episodes get
the instruction "explore the scene". Episodes alternate between the given tasks' scenes.

Per-episode diagnostics (env steps so far, how far objects moved) go to OUT/exploration_log.json.

  python hpc/collect_exploration.py --method icm --suite libero_object --task-ids 0,1 \
      --steps 20000 --demo-root DATA_ROOT --out runs/exploration/icm_libero_object_tasks0-1
"""

import argparse
import faulthandler
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EXPLORE_TASK = "explore the scene"
DEFAULT_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


# ----------------------------------------------------------------------------- environments


class LiberoScene:
    """One LIBERO task scene through LeRobot's wrapper, returning dataset-format observations."""

    def __init__(self, suite_name: str, task_id: int, width: int, height: int, seed: int):
        from libero.libero import benchmark
        from lerobot.sim.integrations.libero import LiberoEnv

        suite = benchmark.get_benchmark_dict()[suite_name]()
        self.env = LiberoEnv(
            task_suite=suite, task_id=task_id, task_suite_name=suite_name,
            obs_type="pixels_agent_pos", observation_width=width, observation_height=height,
            init_states=True, episode_index=seed,
        )
        self.seed = seed
        self.n_resets = 0

    def _obs(self, raw):
        from lerobot.sim.integrations.libero import canonical_observation

        frame = canonical_observation(raw)
        qpos = np.asarray(self.env._env.sim.data.qpos, dtype=np.float32).copy()
        return frame, qpos

    def reset(self):
        raw, _ = self.env.reset(seed=self.seed + self.n_resets)
        self.n_resets += 1
        return self._obs(raw)

    def step(self, action):
        raw, _, terminated, _, info = self.env.step(np.asarray(action, dtype=np.float64))
        frame, qpos = self._obs(raw)
        return frame, qpos, bool(info.get("is_success", False))

    def close(self):
        self.env.close()


class FakeScene:
    """Stand-in with the same interface, for testing this script without LIBERO."""

    def __init__(self, suite_name, task_id, width, height, seed):
        self.rng = np.random.default_rng(seed + task_id)
        self.w, self.h = width, height

    def _obs(self):
        img = np.zeros((self.h, self.w, 3), np.uint8)
        img[..., 0] = np.clip(self.q[0] * 40 + 128, 0, 255)
        frame = {"observation.images.image": img, "observation.images.image2": img.copy(),
                 "observation.state": np.concatenate([self.q[:6], [0.04, -0.04]]).astype(np.float32)}
        return frame, self.q.astype(np.float32).copy()

    def reset(self):
        self.q = self.rng.normal(size=16).astype(np.float32) * 0.1
        return self._obs()

    def step(self, action):
        self.q[:7] += 0.05 * np.asarray(action)
        self.q[9:12] += 0.01 * (np.abs(self.q[:3]).sum() > 1.0)  # "objects" move only far away
        return (*self._obs(), False)

    def close(self):
        pass


# ----------------------------------------------------------------------------- explorers


class RunningNorm:
    def __init__(self, dim):
        self.n, self.mean, self.m2 = 1e-4, np.zeros(dim), np.ones(dim)

    def update(self, x):
        x = np.atleast_2d(x)
        for row in x:
            self.n += 1
            d = row - self.mean
            self.mean += d / self.n
            self.m2 += d * (row - self.mean)

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.m2 / self.n + 1e-8), -5, 5).astype(np.float32)


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class RandomExplorer:
    def __init__(self, obs_dim, act_dim, rng, **_):
        self.rng, self.act_dim = rng, act_dim

    def act(self, obs):
        a = self.rng.uniform(-1, 1, self.act_dim)
        a[-1] = self.rng.choice([-1.0, 1.0])
        return a, {}

    def observe(self, obs, action, next_obs, done, extra):
        return 0.0


class ICMExplorer:
    """PPO on the ICM intrinsic reward, acting once per held macro-action."""

    def __init__(self, obs_dim, act_dim, rng, rollout=512, feat_dim=64, eta=1.0, beta=0.2,
                 lr=3e-4, epochs=4, minibatch=128, gamma=0.99, lam=0.95, clip=0.2, seed=0):
        torch.manual_seed(seed)
        self.rng, self.act_dim = rng, act_dim
        self.norm = RunningNorm(obs_dim)
        self.pi, self.v = mlp(obs_dim, act_dim), mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.enc = mlp(obs_dim, feat_dim)
        self.inv = mlp(2 * feat_dim, act_dim)
        self.fwd = mlp(feat_dim + act_dim, feat_dim)
        self.opt_pi = torch.optim.Adam([*self.pi.parameters(), *self.v.parameters(), self.log_std], lr=lr)
        self.opt_icm = torch.optim.Adam([*self.enc.parameters(), *self.inv.parameters(), *self.fwd.parameters()], lr=lr)
        self.eta, self.beta, self.rollout, self.epochs, self.mb = eta, beta, rollout, epochs, minibatch
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.buf = []
        self.ret_rms, self.ret = RunningNorm(1), 0.0
        self.stats = []

    def _dist(self, o):
        return torch.distributions.Normal(self.pi(o), self.log_std.exp())

    def act(self, obs):
        self.norm.update(obs)
        o = torch.as_tensor(self.norm(obs))[None]
        with torch.no_grad():
            d = self._dist(o)
            u = d.sample()
            logp = d.log_prob(u).sum(-1)
            v = self.v(o)[0, 0]
        a = np.clip(u[0].numpy(), -1, 1)
        a[-1] = 1.0 if a[-1] > 0 else -1.0
        return a, {"o": o[0], "u": u[0], "logp": float(logp), "v": float(v)}

    def _intrinsic(self, o, a, o2):
        with torch.no_grad():
            f, f2 = self.enc(o), self.enc(o2)
            pred = self.fwd(torch.cat([f, a], -1))
            return (0.5 * self.eta * ((pred - f2) ** 2).sum(-1))

    def observe(self, obs, action, next_obs, done, extra):
        o2 = torch.as_tensor(self.norm(next_obs))
        r = float(self._intrinsic(extra["o"][None], torch.as_tensor(action, dtype=torch.float32)[None], o2[None])[0])
        self.buf.append({**extra, "a": torch.as_tensor(action, dtype=torch.float32), "o2": o2, "r": r, "done": done})
        if len(self.buf) >= self.rollout:
            self._update()
        return r

    def _update(self):
        b = self.buf
        o = torch.stack([x["o"] for x in b]); o2 = torch.stack([x["o2"] for x in b])
        u = torch.stack([x["u"] for x in b]); a = torch.stack([x["a"] for x in b])
        logp_old = torch.tensor([x["logp"] for x in b]); v = torch.tensor([x["v"] for x in b])
        r = np.array([x["r"] for x in b]); done = np.array([x["done"] for x in b], dtype=np.float32)
        # Normalize the intrinsic reward by a running std of its discounted return.
        for ri in r:
            self.ret = self.ret * self.gamma + ri
            self.ret_rms.update(np.array([self.ret]))
        r = r / np.sqrt(self.ret_rms.m2 / self.ret_rms.n + 1e-8)[0]
        with torch.no_grad():
            v_last = self.v(o2[-1:])[0, 0].item()
        adv, last = np.zeros(len(b), np.float32), 0.0
        for t in reversed(range(len(b))):
            nv = v_last if t == len(b) - 1 else v[t + 1].item()
            nonterm = 1.0 - done[t]
            delta = r[t] + self.gamma * nv * nonterm - v[t].item()
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        adv = torch.as_tensor(adv); ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = np.arange(len(b))
        for _ in range(self.epochs):
            self.rng.shuffle(idx)
            for s in range(0, len(b), self.mb):
                j = torch.as_tensor(idx[s:s + self.mb])
                d = self._dist(o[j])
                ratio = (d.log_prob(u[j]).sum(-1) - logp_old[j]).exp()
                pl = -torch.min(ratio * adv[j], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[j]).mean()
                vl = ((self.v(o[j])[:, 0] - ret[j]) ** 2).mean()
                self.opt_pi.zero_grad(); (pl + 0.5 * vl).backward(); self.opt_pi.step()
                f, f2 = self.enc(o[j]), self.enc(o2[j])
                inv_l = ((self.inv(torch.cat([f, f2], -1)) - a[j]) ** 2).mean()
                fwd_l = 0.5 * ((self.fwd(torch.cat([f, a[j]], -1)) - f2.detach()) ** 2).sum(-1).mean()
                self.opt_icm.zero_grad(); ((1 - self.beta) * inv_l + self.beta * fwd_l).backward(); self.opt_icm.step()
        self.stats.append({"mean_intrinsic": float(np.mean([x["r"] for x in b])), "fwd_loss": float(fwd_l),
                           "inv_loss": float(inv_l), "policy_std": float(self.log_std.exp().mean())})
        self.buf = []


# ----------------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["random", "icm"], required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--task-ids", required=True)
    p.add_argument("--steps", type=int, required=True, help="total env steps over all tasks")
    p.add_argument("--episode-len", type=int, default=200)
    p.add_argument("--hold", type=int, default=5, help="env steps each sampled action is held")
    p.add_argument("--rollout", type=int, default=512, help="ICM: decisions per PPO update")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demo-root", required=True, help="local demo dataset; features and fps are copied from it")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fake-env", action="store_true", help="test without LIBERO")
    args = p.parse_args()

    # The explorer networks are tiny: one thread avoids OpenMP thread contention inside a Slurm
    # CPU allocation. Every 5 minutes, dump all Python stacks to the log so a stall shows where.
    torch.set_num_threads(1)
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((Path(args.demo_root) / "meta" / "info.json").read_text())
    features = {k: {kk: v[kk] for kk in ("dtype", "shape", "names") if kk in v}
                for k, v in info["features"].items() if k not in DEFAULT_KEYS}
    img_keys = [k for k, v in features.items() if v["dtype"] in ("video", "image")]
    h, w = features[img_keys[0]]["shape"][:2]
    act_dim = features["action"]["shape"][0]

    Scene = FakeScene if args.fake_env else LiberoScene
    task_ids = [int(t) for t in args.task_ids.split(",")]
    scenes = [Scene(args.suite, t, w, h, args.seed) for t in task_ids]
    rng = np.random.default_rng(args.seed)

    if args.out.exists():
        shutil.rmtree(args.out)
    ds = LeRobotDataset.create(f"local/{args.out.name}", fps=info["fps"], features=features, root=args.out,
                               robot_type=info.get("robot_type"), use_videos=any(v["dtype"] == "video" for v in features.values()))

    explorer = None
    log, steps, ep, t0 = [], 0, 0, time.time()
    while steps < args.steps:
        scene = scenes[ep % len(scenes)]
        frame, qpos = scene.reset()
        if explorer is None:
            obs_dim = len(frame["observation.state"]) + len(qpos)
            cls = ICMExplorer if args.method == "icm" else RandomExplorer
            explorer = cls(obs_dim, act_dim, rng, rollout=args.rollout, seed=args.seed)
        q0 = qpos.copy()
        obs = np.concatenate([frame["observation.state"], qpos])
        ep_len = min(args.episode_len, args.steps - steps)
        r_sum, max_obj_move, successes = 0.0, 0.0, 0
        for t in range(ep_len):
            if t % 50 == 0:
                print(f"[{args.method}]   step {t}/{ep_len} of episode {ep + 1} ({time.time() - t0:.0f} s)", flush=True)
            if t % args.hold == 0:
                action, extra = explorer.act(obs)
                start_obs = obs
            ds.add_frame({**{k: frame[k] for k in features if k != "action"},
                          "action": action.astype(np.float32), "task": EXPLORE_TASK})
            frame, qpos, success = scene.step(action)
            successes += success
            obs = np.concatenate([frame["observation.state"], qpos])
            max_obj_move = max(max_obj_move, float(np.abs(qpos[9:] - q0[9:]).max()) if len(qpos) > 9 else 0.0)
            if (t + 1) % args.hold == 0 or t == ep_len - 1:
                r_sum += explorer.observe(start_obs, action, obs, t == ep_len - 1, extra)
        ds.save_episode()
        steps += ep_len
        log.append({"episode": ep, "task_id": task_ids[ep % len(task_ids)], "length": ep_len,
                    "env_steps_after": steps, "intrinsic_reward": r_sum,
                    "max_object_qpos_change": max_obj_move, "task_success_steps": successes})
        ep += 1
        print(f"[{args.method}] episode {ep} task {log[-1]['task_id']} steps {steps}/{args.steps} "
              f"obj_move {max_obj_move:.3f} r_int {r_sum:.2f} ({steps / (time.time() - t0):.0f} steps/s)", flush=True)
    ds.finalize()
    for s in scenes:
        s.close()
    summary = {"method": args.method, "suite": args.suite, "task_ids": task_ids, "steps": steps,
               "episodes": ep, "hold": args.hold, "episode_len": args.episode_len, "seed": args.seed,
               "seconds": round(time.time() - t0), "episodes_log": log,
               "icm_updates": getattr(explorer, "stats", [])}
    (args.out / "exploration_log.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {ep} episodes, {steps} steps to {args.out}")


if __name__ == "__main__":
    main()
