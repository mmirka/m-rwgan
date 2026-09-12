#!/usr/bin/env bash
# Re-bake every derived artifact from the raw archives under figures/data/.
#
# You only need this if you are changing a bake script or re-deriving from a
# different archive: the artifacts it writes are committed, so a plain clone
# already has them and rebuilds every figure without downloading anything.
#
# Requires the git-ignored raw inputs (~1.2 GB) — see figures/data/README.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"

"$PY" bake/bake_dataset_summary.py
"$PY" bake/bake_dse_front.py
"$PY" bake/bake_history_classes.py
"$PY" bake/bake_loss_curves.py

echo
echo "derived artifacts now under figures/data/derived/:"
find data/derived -name "*.npz" -print0 | du -ch --files0-from=- | tail -1
