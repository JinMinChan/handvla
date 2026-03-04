#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="dry-run"
if [[ "${1:-}" == "--execute" ]]; then
  MODE="execute"
fi

# Keep exactly one representative video per algorithm family.
KEEP_VIDEOS=(
  "codex/logs/full_joint_diverse_rollout_demo.mp4"        # full_joint baseline
  "codex/logs/tcp12_cmd_next_palm_local_rollout_demo_20260303.mp4"  # tcp12 failure representative
  "codex/logs/synergy_k4_selected_rollout_demo_20260304.mp4"        # synergy-k4 representative
  "codex/logs/finger_ik_260219_164246.mp4"               # IK feasibility representative
)

in_keep() {
  local target="$1"
  local x
  for x in "${KEEP_VIDEOS[@]}"; do
    [[ "$target" == "$x" ]] && return 0
  done
  return 1
}

act_rm() {
  local p="$1"
  if [[ "$MODE" == "execute" ]]; then
    rm -f -- "$p"
    echo "[DELETE] $p"
  else
    echo "[DRY]    $p"
  fi
}

echo "Mode: $MODE"
while IFS= read -r f; do
  # prune only videos/images in logs; keep summaries (.json/.md)
  case "$f" in
    *.mp4|*.png)
      if ! in_keep "$f"; then
        act_rm "$f"
      fi
      ;;
  esac
done < <(find codex/logs -maxdepth 1 -type f | sort)

echo "Remaining media files:"
find codex/logs -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.png' \) | sort

echo "Current logs size:"
du -sh codex/logs

if [[ "$MODE" == "dry-run" ]]; then
  echo "Dry-run complete. Re-run with --execute to apply deletions."
else
  echo "Prune complete."
fi
