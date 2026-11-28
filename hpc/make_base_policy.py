"""Make a copy of smolvla_base whose input/output features match LIBERO.

Why: when lerobot-train starts from a pretrained policy, it keeps that policy's saved
input_features. smolvla_base was pretrained with a 6-dim state and cameras camera1-3, while
LIBERO has an 8-dim state and cameras image/image2. Training still ran (SmolVLA pads the state
to 32 dims), but the checkpoint then expected the wrong state shape at eval time. That kind of
mismatch is the silent 0%-success trap from the proposal.

So this copies smolvla_base's files into OUT, and rewrites config.json so that its
input_features and output_features are exactly those of the official LIBERO fine-tune
(HuggingFaceVLA/smolvla_libero). It checks that every one of those features exists in the
demo dataset with the same shape. Weights are linked, not copied.

  python hpc/make_base_policy.py BASE_DIR REFERENCE_DIR DATASET_ROOT OUT_DIR
"""

import json
import os
import shutil
import sys
from pathlib import Path


def features(cfg: dict) -> dict:
    return {**cfg["input_features"], **cfg["output_features"]}


def main() -> None:
    base, ref, data_root, out = (Path(a) for a in sys.argv[1:5])
    base_cfg = json.loads((base / "config.json").read_text())
    ref_cfg = json.loads((ref / "config.json").read_text())
    ds_features = json.loads((data_root / "meta" / "info.json").read_text())["features"]

    errors = []
    for key, ft in features(ref_cfg).items():
        if key not in ds_features:
            errors.append(f"{key} is used by {ref.name} but missing from the dataset")
            continue
        ds_shape = list(ds_features[key]["shape"])
        if ds_features[key].get("dtype") in ("video", "image") and ds_shape[-1] == 3:
            ds_shape = [3] + ds_shape[:-1]  # dataset stores HWC, policies use CHW
        if list(ft["shape"]) != ds_shape:
            errors.append(f"{key}: reference shape {ft['shape']} != dataset shape {ds_shape}")
    if errors:
        sys.exit("Feature check failed:\n  " + "\n  ".join(errors))

    print("smolvla_base features (replaced):")
    for k, v in features(base_cfg).items():
        print(f"  {k}: {v['shape']}")
    print("LIBERO features (from the reference checkpoint, checked against the dataset):")
    for k, v in features(ref_cfg).items():
        print(f"  {k}: {v['shape']}")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in base.iterdir():
        if f.name != "config.json":
            os.symlink(f.resolve(), out / f.name)
    base_cfg["input_features"] = ref_cfg["input_features"]
    base_cfg["output_features"] = ref_cfg["output_features"]
    (out / "config.json").write_text(json.dumps(base_cfg, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
