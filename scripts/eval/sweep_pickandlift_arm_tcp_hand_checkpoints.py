#!/usr/bin/env python3
"""Evaluate multiple end-to-end arm+hand Octo checkpoints and rank them by rollout."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROLLOUT_SCRIPT = REPO_ROOT / "scripts" / "eval" / "rollout_pickandlift_arm_tcp_hand_octo.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pick-and-lift arm+hand rollout evaluator over a set of saved checkpoints "
            "and rank them by closed-loop success."
        )
    )
    parser.add_argument("--run-dir", type=str, required=True, help="Training run directory containing checkpoints.")
    parser.add_argument("--basis-path", type=str, required=True, help="Hand synergy basis path.")
    parser.add_argument("--episodes", type=int, default=1, help="Rollout episodes per checkpoint.")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation RNG seed.")
    parser.add_argument(
        "--include",
        type=str,
        default="checkpoint_*,best_model,final_model,best_model_snapshot_live_step*",
        help="Comma-separated glob patterns inside run-dir.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="",
        help="Optional output JSON path. Defaults under codex/logs/json.",
    )
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Stop scanning after the first checkpoint with success_rate > 0.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many ranked checkpoints to keep in the summary.",
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


def _default_summary_path(run_dir: Path) -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return REPO_ROOT / "codex" / "logs" / "json" / f"{run_dir.name}_checkpoint_sweep_{ts}.json"


def _collect_checkpoints(run_dir: Path, include_patterns: str) -> list[Path]:
    found: dict[str, Path] = {}
    for pattern in [p.strip() for p in include_patterns.split(",") if p.strip()]:
        for path in sorted(run_dir.glob(pattern)):
            if path.is_dir():
                found[path.name] = path.resolve()
    return [found[name] for name in sorted(found.keys())]


def _episode_means(summary: dict[str, Any]) -> dict[str, float]:
    eps = summary.get("episodes_data", [])
    if not isinstance(eps, list) or not eps:
        return {
            "success_rate": float(summary.get("success_rate", 0.0) or 0.0),
            "mean_reached": 0.0,
            "mean_grasp": 0.0,
            "mean_lift": 0.0,
            "mean_contacts": 0.0,
            "mean_dz_max": 0.0,
            "mean_approach_err": 1e9,
        }
    def _avg(key: str) -> float:
        vals = []
        for ep in eps:
            if isinstance(ep, dict):
                vals.append(float(ep.get(key, 0.0)))
        return sum(vals) / max(len(vals), 1)
    return {
        "success_rate": float(summary.get("success_rate", 0.0) or 0.0),
        "mean_reached": _avg("reached"),
        "mean_grasp": _avg("grasp_acquired"),
        "mean_lift": _avg("lift_acquired"),
        "mean_contacts": _avg("best_contacts"),
        "mean_dz_max": _avg("object_dz_max"),
        "mean_approach_err": _avg("approach_min_err"),
    }


def _rank_key(record: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    m = record["metrics"]
    return (
        float(m["success_rate"]),
        float(m["mean_lift"]),
        float(m["mean_grasp"]),
        float(m["mean_reached"]),
        float(m["mean_dz_max"]),
        -float(m["mean_approach_err"]),
    )


def main() -> None:
    args, rollout_extra = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    basis_path = Path(args.basis_path).expanduser().resolve()
    summary_path = (
        Path(args.summary_path).expanduser().resolve() if args.summary_path else _default_summary_path(run_dir)
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoints = _collect_checkpoints(run_dir, args.include)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched in {run_dir} with patterns: {args.include}")

    results: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None

    for ckpt in checkpoints:
        out_json = summary_path.parent / f"{run_dir.name}_{ckpt.name}_eval.json"
        cmd = [
            sys.executable,
            str(ROLLOUT_SCRIPT),
            "--model-path",
            str(ckpt),
            "--basis-path",
            str(basis_path),
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--no-viewer",
            "--save-json",
            str(out_json),
        ] + rollout_extra
        print("[sweep] evaluating", ckpt)
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        record: dict[str, Any] = {
            "checkpoint": str(ckpt),
            "save_json": str(out_json),
            "returncode": int(proc.returncode),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
        if proc.returncode == 0 and out_json.exists():
            summary = json.loads(out_json.read_text(encoding="utf-8"))
            record["metrics"] = _episode_means(summary)
            record["success_rate"] = float(summary.get("success_rate", 0.0) or 0.0)
            record["episodes"] = int(summary.get("episodes", 0) or 0)
        else:
            record["metrics"] = _episode_means({})
            record["success_rate"] = 0.0
            record["episodes"] = 0
        results.append(record)
        if best_record is None or _rank_key(record) > _rank_key(best_record):
            best_record = record
        if args.stop_on_success and record["success_rate"] > 0.0:
            break

    ranked = sorted(results, key=_rank_key, reverse=True)
    payload = {
        "run_dir": str(run_dir),
        "basis_path": str(basis_path),
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "extra_args": rollout_extra,
        "checked": len(results),
        "best": best_record,
        "ranked": ranked[: max(int(args.top_k), 1)],
        "all_results": results,
        "saved_at": datetime.now().isoformat(),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] wrote {summary_path}")
    if best_record is not None:
        print(f"[best] {best_record['checkpoint']} success_rate={best_record['success_rate']:.3f}")


if __name__ == "__main__":
    main()
