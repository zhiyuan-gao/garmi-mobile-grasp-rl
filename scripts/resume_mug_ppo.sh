#!/usr/bin/env bash
# Run on the destination machine after restoring and verifying the resume bundle.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
num_envs="${GARMI_RL_NUM_ENVS:-4096}"
final_update="${GARMI_RL_FINAL_UPDATE:-2500}"
output="${GARMI_RL_RESUME_OUT:-runs/mug-ppo-resumed}"
[[ "$num_envs" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid environment count' >&2; exit 1; }
[[ "$final_update" =~ ^[0-9]+$ ]] && ((final_update >= 2500 && final_update % 500 == 0)) || {
  echo 'Final update must be 2500 or a later multiple of 500' >&2; exit 1;
}
[[ "$output" == runs/* && "$output" != *..* ]] || { echo 'Output must be a new runs/... directory' >&2; exit 1; }
python3 scripts/export_ppo_resume.py verify --root "$project_dir"
mkdir "$output"  # Refuse to overwrite any prior run, including another running queue.
checkpoint=/workspace/runs/wrist-ppo-v6/mug/ppo-2000/checkpoints/update_2000.pt
config=/workspace/runs/wrist-ppo-v6/mug/ppo.yaml
base=/workspace/runs/wrist-imitation-v6/mug/round-03-bc/last.pt
check_log() {
  python3 - "$1" "$2" <<'PY'
import sys
from pathlib import Path
text=Path(sys.argv[1]).read_text(errors='replace')
assert sys.argv[2] in text, 'Missing completion marker'
bad=[x for x in ('Traceback (most recent call last)', 'CUDA error:', 'out of memory',
                'buffer overflow', 'PhysX error', 'increase PxgDynamicsMemoryConfig')
     if x.lower() in text.lower()]
assert not bad, bad
PY
}
state() {
  python3 - "$output/status.json" "$1" "$2" "$num_envs" "$final_update" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps(dict(status=sys.argv[2],target_stage=int(sys.argv[3]),
    num_envs=int(sys.argv[4]),final_update=int(sys.argv[5]),updated_unix=time.time()),indent=2)+'\n')
tmp.replace(p)
PY
}
target=2000
trap 'state failed "$target"' ERR
for ((target=2500; target<=final_update; target+=500)); do
  stage="$output/ppo-$target"
  state training "$target"
  bash scripts/rl.sh /workspace/rl/train_trial.py --headless \
    --config "$config" --out "/workspace/$stage" --num_envs "$num_envs" \
    --updates "$target" --eval-updates "$target" --skip-evaluation --seed 2026102002 \
    --resume "$checkpoint" --allow-env-resize > "$output/ppo-$target.log" 2>&1
  check_log "$output/ppo-$target.log" TRIAL_DONE
  checkpoint="/workspace/$stage/checkpoints/update_$(printf '%04d' "$target").pt"
  state auditing "$target"
  bash scripts/rl.sh /workspace/rl/audit_ppo_transfer.py --bc "$base" \
    --checkpoint "$checkpoint" --stage "/workspace/$stage" --expected-update "$target" \
    --out "/workspace/$output/audit-$target.json" > "$output/audit-$target.log" 2>&1
  check_log "$output/audit-$target.log" PPO_TRANSFER_AUDIT_PASSED
  state evaluating "$target"
  bash scripts/rl.sh /workspace/rl/evaluate_kitchen.py --headless \
    --config "$config" --checkpoint "$checkpoint" --num_envs 80 --seed 12345 \
    --require-strict-success --out "/workspace/$output/evaluation-$target" \
    > "$output/evaluation-$target.log" 2>&1
  check_log "$output/evaluation-$target.log" EVALUATION_DONE
  python3 - "$output/evaluation-$target/summary.json" "${checkpoint#/workspace/}" <<'PY'
import hashlib,json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d['status']=='complete' and d['completed']==80 and d['require_strict_success']
assert d['checkpoint_sha256']==hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
assert d['evaluation_cases_sha256']==hashlib.sha256(Path('runs/wrist-ppo-v6/mug/evaluation-cases.json').read_bytes()).hexdigest()
assert all(r['strict_success_verified'] for r in d['results'] if r['reason']=='success')
PY
  echo "Completed update $target and its strict evaluation"
done
state complete "$final_update"
