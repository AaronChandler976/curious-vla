"""Assemble the 2-minute demo video (silent, captioned; add a voice-over on top if you like).

  python video/make_video.py --runs runs --figs slides/figures --out video/demo.mp4

Scenes: title, the problem, the pipeline, real LIBERO rollouts side by side (a demos-only policy
that succeeds next to an exploration-mix policy that fails on the same initial state), the main
results, the diagnosis and the takeaway. Rollout clips come from the evaluation videos that
lerobot-eval saved under runs/<run>/eval/videos/. Figures are the talk's PDFs, rasterized with
pdftoppm. Output: 1280x720, 30 fps, mp4 (H.264 if the OpenCV build has it, else MPEG-4 part 2).
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
BG, INK, MUTED, ACCENT = (247, 250, 251), (43, 35, 29), (112, 98, 90), (12, 65, 194)   # BGR
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def text_img(img, lines, top=90, size=40, bold_first=True, color=INK, left=80):
    """Draw lines of text with PIL (anti-aliased) onto a BGR image."""
    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    y = top
    for i, ln in enumerate(lines):
        f = ImageFont.truetype(BOLD if (i == 0 and bold_first) else FONT, size if i == 0 else int(size * 0.7))
        d.text((left, y), ln, font=f, fill=color[::-1])
        y += int((size if i == 0 else size * 0.7) * 1.45)
    return np.asarray(pil)[:, :, ::-1].copy()


def card(lines, seconds, size=44):
    img = np.full((H, W, 3), BG, np.uint8)
    img = text_img(img, lines, top=200 if len(lines) <= 3 else 120, size=size)
    return [img] * int(seconds * FPS)


def figure_card(pdf, title, caption, seconds, tmp):
    png = Path(tmp) / (Path(pdf).stem + ".png")
    subprocess.run(["pdftoppm", "-r", "220", "-png", "-singlefile", str(pdf), str(png.with_suffix(""))], check=True)
    fig = cv2.imread(str(png))
    s = min((W - 80) / fig.shape[1], (H - 210) / fig.shape[0])
    fig = cv2.resize(fig, (int(fig.shape[1] * s), int(fig.shape[0] * s)), interpolation=cv2.INTER_AREA)
    img = np.full((H, W, 3), BG, np.uint8)
    x0, y0 = (W - fig.shape[1]) // 2, 105
    img[y0:y0 + fig.shape[0], x0:x0 + fig.shape[1]] = fig
    img = text_img(img, [title], top=30, size=36)
    img = text_img(img, [caption], top=H - 75, size=26, bold_first=False, color=MUTED)
    return [img] * int(seconds * FPS)


def read_clip(path):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames


def rollout_scene(left, right, labels, title, seconds):
    a, b = read_clip(left), read_clip(right)
    n = int(seconds * FPS)
    out = []
    side = 500
    for i in range(n):
        img = np.full((H, W, 3), BG, np.uint8)
        for j, (clip, lab, col) in enumerate(zip((a, b), labels, ((122, 160, 27), ACCENT))):
            f = clip[min(int(i * len(clip) / n), len(clip) - 1)]
            f = cv2.resize(f, (side, side))
            x0 = 90 + j * (side + 100)
            img[130:130 + side, x0:x0 + side] = f
            cv2.rectangle(img, (x0 - 3, 127), (x0 + side + 2, 130 + side + 2), col, 3)
            img = text_img(img, [lab], top=130 + side + 12, size=26, left=x0, bold_first=True)
        img = text_img(img, [title], top=40, size=36)
        out.append(img)
    return out


def pick_episode(run_dir, task, want_success):
    info = json.loads((run_dir / "eval" / "eval_info.json").read_text())
    t = next(t for t in info["per_task"] if t["task_id"] == task)
    for i, s in enumerate(t["metrics"]["successes"][:10]):
        p = run_dir / "eval" / "videos" / f"libero_object_{task}" / f"eval_episode_{i}.mp4"
        if bool(s) == want_success and p.exists():
            return p, i
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--figs", type=Path, default=Path("slides/figures"))
    ap.add_argument("--good", default="demos_libero_object_tasks0-1_n40_s0")
    ap.add_argument("--bad", default="mix-icm-20000_libero_object_tasks0-1_n40_s0")
    ap.add_argument("--out", type=Path, default=Path("video/demo.mp4"))
    a = ap.parse_args()
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        frames += card(["Can a robot learn from playing on its own?",
                        "Exploration data for fine-tuning a small vision-language-action model",
                        "Aaron Chandler  ·  CMPE 249  ·  SJSU"], 5)
        frames += card(["Demos are expensive. Play is free.",
                        "VLA robot policies learn from teleoperated demonstrations.",
                        "A robot can explore by itself at no cost, but that data has no labels.",
                        "Question: does curiosity-driven play help more than random play,",
                        "or than no play at all?"], 10)
        frames += card(["The experiment",
                        "SmolVLA (450M params) on 2 LIBERO tasks, 10 / 25 / 40 demos per task",
                        "+ random play  or  + curiosity (ICM) play, 5k or 20k steps",
                        "Same training steps, same data path, 10 seeds per setting",
                        "464 fine-tunes on SJSU's A100/H100 cluster"], 11)
        good, gi = pick_episode(a.runs / a.good, 0, True)
        bad, bi = pick_episode(a.runs / a.bad, 0, False)
        if good and bad:
            frames += rollout_scene(good, bad, ["Demos only: grasps and places", "Demos + curiosity play: hovers, never grasps"],
                                    "Same task, same start: put the alphabet soup in the basket", 14)
        good1, _ = pick_episode(a.runs / a.good, 1, True)
        bad1, _ = pick_episode(a.runs / a.bad, 1, False)
        if good1 and bad1:
            frames += rollout_scene(good1, bad1, ["Demos only", "Demos + curiosity play"], "Second task: the cream cheese", 10)
        frames += figure_card(a.figs / "c1_main.pdf", "Result: adding play never beats demos only",
                              "Mean of 10 seeds with 95% intervals; play costs up to 19 points", 13, tmp)
        frames += figure_card(a.figs / "c2_replicates.pdf", "Curiosity vs random flips sign, and the data draw matters as much",
                              "Each bar is one independent collection of exploration data", 13, tmp)
        frames += card(["Why it fails",
                        "SmolVLA only trains its action expert; the vision-language model is frozen.",
                        "Play actions can't be predicted from what the robot sees (R² ≈ 0 vs 0.42).",
                        "So imitating them teaches noise. How data is used matters more",
                        "than which explorer collected it."], 12)
        frames += card(["Takeaway",
                        "Don't mix task-free play into VLA fine-tuning with an imitation loss.",
                        "Next: inverse-dynamics pretraining, trained curiosity, unfrozen VLM.",
                        "github.com/AaronChandler976/curious-vla"], 8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*codec), FPS, (W, H))
        if vw.isOpened():
            break
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"wrote {a.out}: {len(frames) / FPS:.1f} s, codec {codec}")


if __name__ == "__main__":
    main()
