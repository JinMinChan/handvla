#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="dry-run"
if [[ "${1:-}" == "--execute" ]]; then
  MODE="execute"
fi

# Keep only artifacts needed for ongoing comparison + reproducibility.
KEEP_DATASETS=(
  "mustard_grasp_full_joint_diverse"
  "mustard_grasp_oxe_full_joint_diverse"
  "mustard_grasp_synergy_k4_diverse_k236"
  "mustard_grasp_oxe_synergy_k4_diverse_k236"
  "synergy_basis"
  "synergy_basis_diverse"
  "synergy_basis_diverse_k236"
)

KEEP_MODELS=(
  "mustard_octo_overfit_full_joint_diverse"
  "mustard_octo_overfit_synergy_k4_selected"
)

EXTRA_DELETE=(
  "codex/tmp_arxiv_2511"
  "codex/tmp_joint16_cmd_dataset"
  "codex/tmp_tcp12_cmd_dataset"
  "codex/tmp_tcp12_cmd_next_dataset"
  "codex/tmp_mustard12"
  "__pycache__"
  "env/__pycache__"
)

in_keep_list() {
  local item="$1"
  shift
  local keep
  for keep in "$@"; do
    if [[ "$item" == "$keep" ]]; then
      return 0
    fi
  done
  return 1
}

act_rm() {
  local target="$1"
  if [[ "$MODE" == "execute" ]]; then
    rm -rf -- "$target"
    echo "[DELETE] $target"
  else
    echo "[DRY]    $target"
  fi
}

prune_under() {
  local base="$1"
  shift
  local keeps=("$@")

  [[ -d "$base" ]] || return 0

  local p name
  while IFS= read -r p; do
    name="$(basename "$p")"
    if ! in_keep_list "$name" "${keeps[@]}"; then
      act_rm "$p"
    fi
  done < <(find "$base" -mindepth 1 -maxdepth 1 -type d | sort)
}

echo "Mode: $MODE"
echo "Pruning dataset/ ..."
prune_under "dataset" "${KEEP_DATASETS[@]}"

echo "Pruning models/ ..."
prune_under "models" "${KEEP_MODELS[@]}"

echo "Deleting temp/cache paths ..."
for p in "${EXTRA_DELETE[@]}"; do
  if [[ -e "$p" ]]; then
    act_rm "$p"
  fi
done

echo "Remaining top-level sizes:"
du -sh codex dataset models 2>/dev/null || true

echo "Remaining dataset dirs:"
find dataset -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort || true

echo "Remaining model dirs:"
find models -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort || true

if [[ "$MODE" == "dry-run" ]]; then
  echo "Dry-run complete. Re-run with --execute to apply deletions."
else
  echo "Cleanup complete."
fi
