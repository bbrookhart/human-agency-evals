#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in this shell; never commit it.}"
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY in this shell; never commit it.}"
: "${CALIBRATION_APPROVED_ESTIMATE_USD:?Set CALIBRATION_APPROVED_ESTIMATE_USD after reviewing the preflight estimate.}"

run_id="${CALIBRATION_RUN_ID:-calibration-001}"
config="configs/experiment_calibration.yaml"
preflight_path="runs/${run_id}-preflight.json"
annotation_path="annotations/${run_id}.csv"

mkdir -p runs annotations
uv sync --locked --extra providers
uv run human-agency-evals preflight --config "$config" > "$preflight_path"

uv run python - "$preflight_path" "$CALIBRATION_APPROVED_ESTIMATE_USD" <<'PY'
import json
import sys

path, approved_raw = sys.argv[1:]
report = json.loads(open(path).read())
approved = float(approved_raw)
estimate = report["estimate"]["estimated_total_usd"]
failed = [
    check
    for check in report["checks"]
    if not check["sdk_available"]
    or not check["credential_present"]
    or not check["pricing_known"]
]
if failed:
    raise SystemExit("Preflight failed: provider SDK, credential, or pricing check is incomplete.")
if estimate is None:
    raise SystemExit("Preflight failed: cost estimate is unknown.")
if estimate > approved:
    raise SystemExit(
        f"Preflight estimate ${estimate:.2f} exceeds approved ceiling ${approved:.2f}."
    )
print(f"Preflight passed: estimated maximum ${estimate:.2f}; approved estimate threshold ${approved:.2f}.")
PY

uv run human-agency-evals run --config "$config" --run-id "$run_id"
uv run human-agency-evals analyze --run "$run_id"
uv run human-agency-evals report --run "$run_id"
uv run human-agency-evals annotate-export \
  --run "$run_id" \
  --output "$annotation_path" \
  --limit 60

echo "Calibration complete: runs/${run_id}/report.md"
echo "Blinded human-rating batch: ${annotation_path}"
