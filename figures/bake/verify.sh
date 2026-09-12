#!/usr/bin/env bash
# Prove the derived artifacts reproduce the figures exactly.
#
#   1. Build every figure from the raw archives, in this working tree.
#   2. Build every figure again in a scratch copy of figures/ that holds only
#      the git-tracked files — i.e. exactly what a plain clone has, derived
#      artifacts included and raw archives absent.
#   3. Compare the two sets byte for byte.
#
# Step 2 is the real claim under test: not "the derived path also works" but
# "a clone with no download rebuilds the same figures". matplotlib's PNG output
# is deterministic here, so any difference at all is a bake bug and fails.
#
# Requires the git-ignored raw inputs (~1.2 GB) — see figures/data/README.md.
# Usage: bake/verify.sh [scratch-dir]
set -euo pipefail

FIGURES="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$FIGURES/.." && pwd)"
PY="${PYTHON:-python}"
WORK="${1:-$(mktemp -d)}"
RAW_OUT="$WORK/from-raw"
CLONE="$WORK/clone"
CLONE_OUT="$WORK/from-derived"

figures=(
  "pareto_figures.py Power uniform"
  "pareto_figures.py Area uniform"
  "pareto_figures.py Power hotspot30-gcn"
  "pareto_figures.py Area hotspot30-gcn"
  "pareto_figures.py Power hotspot30-cnn"
  "pareto_figures.py Area hotspot30-cnn"
  "loss_curves.py --run-prefix data/reward_weight_sweep_8x8_3reward/uniform_3c/multiRWGAN_R-SatGCN_300e_L0.2-d0.05_beta10_SatAndArea_100-0_date08072021_"
  "router_evolution.py --config uniform --type training-progress --reward-ratio Sat100"
  "router_evolution.py --config uniform --type reward-ratio-compare"
  "router_evolution.py --config hotspot --type reward-ratio-compare"
  "router_evolution.py --config hotspot-cnn --type reward-ratio-compare"
  "igd_compare.py"
  "taux_routeurs_date2022.py --style all"
  "router_size_maps.py --traffic hotspot"
  "router_size_maps.py --traffic uniform"
  "best_gene_compare.py"
  "moustache_boxplots.py --figure all"
  "results_analysis_date2022.py --traffic uniform"
  "results_analysis_date2022.py --traffic hotspot"
  "results_analysis_date2022.py --traffic hotspot-area"
)

run_all() {  # run_all <figures-dir> <output-dir>
  local dir="$1" out="$2"
  mkdir -p "$out"
  for spec in "${figures[@]}"; do
    # shellcheck disable=SC2086  # the specs are literal argv, deliberately split
    (cd "$dir" && "$PY" $spec --output-dir "$out" >/dev/null)
  done
}

echo "1/3  building from the raw archives"
run_all "$FIGURES" "$RAW_OUT"

echo "2/3  building from a tracked-files-only copy (what a plain clone has)"
rm -rf "$CLONE"
mkdir -p "$CLONE"
# git ls-files is the definition of "what a clone gets" — it excludes every
# git-ignored raw archive by construction, so nothing here can accidentally
# fall back to one.
(cd "$REPO" && git ls-files -z figures) \
  | (cd "$REPO" && tar --null -cf - --files-from=-) \
  | tar -xf - -C "$CLONE" --strip-components=1
run_all "$CLONE" "$CLONE_OUT"

echo "3/3  comparing"
status=0
n=0
for f in "$RAW_OUT"/*.png; do
  name="$(basename "$f")"
  n=$((n + 1))
  if [[ ! -f "$CLONE_OUT/$name" ]]; then
    echo "  MISSING from the clone build: $name"
    status=1
  elif ! cmp -s "$f" "$CLONE_OUT/$name"; then
    echo "  DIFFERS: $name"
    status=1
  fi
done
for f in "$CLONE_OUT"/*.png; do
  name="$(basename "$f")"
  [[ -f "$RAW_OUT/$name" ]] || { echo "  EXTRA in the clone build: $name"; status=1; }
done

if [[ $status -eq 0 ]]; then
  echo "OK — all $n figures are byte-identical from raw and from derived."
else
  echo "FAILED — see above. Artifacts kept in $WORK"
fi
exit $status
