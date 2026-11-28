#!/bin/bash
# Check, offline, that everything the jobs load is in the cache on this machine. It only writes a
# missing refs/main pointer in the Hugging Face cache; nothing else changes.
#   ssh hpc3 'cd ~/curious-vla && bash hpc/verify_cache.sh'
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source hpc/config.sh
activate_env >/dev/null || exit 1
ok=1
for repo in "$BASE_POLICY" "$REFERENCE_POLICY" $VLM_BACKBONES; do
  pat=""; [[ " $VLM_BACKBONES " == *" $repo "* ]] && pat="$VLM_BACKBONE_FILES"
  if path=$(hf_local_model "$repo" "$pat" 2>/tmp/cvla_verify_err); then echo "OK   model   $repo -> $path"
  else echo "MISSING model $repo: $(tail -1 /tmp/cvla_verify_err)"; ok=0; fi
done
# Each SmolVLA checkpoint names its own backbone in config.json; make sure that is the one we staged.
for repo in "$BASE_POLICY" "$REFERENCE_POLICY"; do
  dir=$(hf_local_model "$repo" 2>/dev/null) || continue
  vlm=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('vlm_model_name',''))" "$dir/config.json")
  if [[ -n "$vlm" && " $VLM_BACKBONES " != *" $vlm "* ]]; then
    echo "MISSING $repo uses backbone $vlm, which is not in VLM_BACKBONES in config.sh"; ok=0
  else
    echo "OK   backbone of $repo is ${vlm:-unset}"
  fi
done
if root=$(dataset_root); then
  echo "OK   dataset $DEMO_DATASET -> $root ($(du -shL "$root" 2>/dev/null | cut -f1))"
  python - "$root" <<'EOF' || ok=0
import sys
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
m = LeRobotDatasetMetadata("HuggingFaceVLA/libero", root=sys.argv[1])
print(f"OK   dataset metadata loads offline: {m.total_episodes} episodes")
EOF
else ok=0; fi
assets="$(python -c 'import libero.libero, os; print(os.path.dirname(libero.libero.__file__))')/assets/scenes/libero_floor_base_style.xml"
if [[ -f "$assets" ]]; then echo "OK   LIBERO assets"; else echo "MISSING LIBERO assets ($assets)"; ok=0; fi
if (( ok )); then echo "ALL CACHED"; else echo "SOMETHING IS MISSING"; exit 1; fi
