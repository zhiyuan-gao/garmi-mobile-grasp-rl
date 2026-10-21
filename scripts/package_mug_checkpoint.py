"""Package the completed mug update-6500 runtime without starting CUDA or Isaac Sim.

The release contains the exact tracked training source, checkpoint pair, profile,
and evaluation evidence. Simulation assets and expert datasets stay in the source
workspace; their hashes are recorded for an optional verified local import.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


MANIFEST = "transfer/mug-ppo-6500/manifest.json"
STAGE = Path("runs/mug-ppo-physical-success/ppo-6500")
REVIEW = Path("runs/mug-physical-success-review")
CHECKPOINT = STAGE / "checkpoints/update_6500.pt"
PROFILE = REVIEW / "ppo.yaml"


def relative_path(value):
    path = Path(value)
    if path.is_absolute():
        path = path.relative_to("/workspace")
    if not path.parts or ".." in path.parts:
        raise ValueError(f"Invalid workspace path: {value}")
    return path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_file(root, relative):
    path = root / relative_path(relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes workspace: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def check_entry(root, relative, entry):
    path = checked_file(root, relative)
    if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
        raise ValueError(f"Hash/size mismatch: {relative}")
    return path


def export(source, output):
    import yaml

    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source).strip():
        raise ValueError("The source tracked runtime must be clean")
    read = lambda p: json.loads((source / p).read_text())
    summary = read(STAGE / "summary.json")
    audit = read("runs/mug-ppo-physical-success/audit-6500.json")
    evaluation = read("runs/mug-ppo-physical-success/evaluation-6500/summary.json")
    assert summary["update"] == 6500 and summary["status"] == "complete"
    assert audit["status"] == "passed" and audit["update"] == 6500
    assert evaluation["status"] == "complete" and evaluation["completed"] == 80
    assert not evaluation["require_strict_success"] and not evaluation["task"]["require_strict_success"]
    assert sha(source / CHECKPOINT) == audit["checkpoint_sha256"] == evaluation["checkpoint_sha256"]
    profile = yaml.safe_load((source / PROFILE).read_text())
    assert profile["garmi_task"]["require_strict_success"] is False
    old = read("transfer/mug-ppo-2000/manifest.json")
    for relative, digest in old["code_sha256"].items():
        assert sha(checked_file(source, relative)) == digest, relative
    selected = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().rstrip("\0").split("\0"))
    selected.update(str(STAGE / name) for name in (
        "summary.json", "progress.json", "latest.json", "task.json", "reward_terms.json",
        "expert_data.json", "initialization_6159.json", "ppo.yaml", "updates.csv",
        "checkpoints/update_6500.pt", "checkpoints/update_6500.resume.pt"))
    selected.update(str(REVIEW / name) for name in (
        "ppo.yaml", "migration.json", "paired-baseline.json", "README.md",
        "baseline-strict-6149/summary.json", "baseline-physical-6149/summary.json",
        "preflight-audit.json", "audit_physical_ppo.py", "migrate_success_rule.py",
        "run_transition.py"))
    selected.update(("runs/mug-resume-physical-success.sh",
                     "runs/mug-ppo-physical-success/audit-6500.json",
                     "runs/mug-ppo-physical-success/evaluation-6500/summary.json",
                     "runs/wrist-ppo-v6/mug/evaluation-cases.json",
                     "runs/wrist-ppo-v6/mug/training-reset-pool.json"))
    dependencies = {p: dict(entry, role="simulation") for p, entry in old["files"].items()
                    if p.startswith(("assets/", "rl/assets/", "resources/"))}

    def dependency(value, role="simulation", expected=None):
        relative = str(relative_path(value))
        entry = old["files"][relative]
        if expected is not None:
            assert entry["sha256"] == expected, relative
        dependencies[relative] = dict(entry, role=role)

    mask = read(relative_path(profile["garmi_task"]["kitchen"]["mask"]))
    for path, digest in mask["sources_sha256"].items():
        dependency(path, expected=digest)
    index = profile["expert_bc"]["index"]
    dependency(index, "expert_training")
    expert = read(relative_path(index))
    for row in expert["demonstrations"] + expert["dagger_label_sets"]:
        dependency(row["file"], "expert_training", row["sha256"])
    # The original BC checkpoint is needed only for the optional transfer audit.
    dependency("runs/wrist-imitation-v6/mug/round-03-bc/last.pt", "expert_training")
    dependencies = {p: e for p, e in dependencies.items() if p not in selected}
    files = {}
    for relative in sorted(selected):
        path = checked_file(source, relative)
        files[relative] = dict(bytes=path.stat().st_size, sha256=sha(path))
    manifest = dict(schema_version=1, object="mug", update=6500,
        runtime_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        runtime_branch="codex/ycb-wrist-v6-ppo2000", checkpoint="/workspace/" + str(CHECKPOINT),
        config="/workspace/" + str(PROFILE), num_envs=4096, seed=2026102002,
        success_criterion="physical_grasp_v1", criterion_changed_at_update=6149,
        evaluation=dict(cases=80, success=62, success_rate=0.775,
                        cases_sha256=evaluation["evaluation_cases_sha256"]),
        expert_trajectories=len(expert["demonstrations"]),
        expert_label_files=len(expert["dagger_label_sets"]), files=files,
        code_sha256=old["code_sha256"], dependencies=dependencies,
        resume_semantics="Restore the model, optimizer, scheduler, normalizers and RNG; start fresh simulation episodes.")
    output.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    (output / "manifest.json").write_bytes(raw)
    archive = output / "mug-ppo-6500.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=3) as tar:
        for relative in sorted(files):
            tar.add(source / relative, arcname=relative, recursive=False)
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    for name in ("update_6500.pt", "update_6500.resume.pt"):
        shutil.copy2(source / STAGE / "checkpoints" / name, output / name)
    names = [archive.name, "manifest.json", "update_6500.pt", "update_6500.resume.pt"]
    (output / "SHA256SUMS").write_text("".join(f"{sha(output / n)}  {n}\n" for n in names))
    print("CHECKPOINT_BUNDLE_EXPORTED", archive, archive.stat().st_size)


def verify(root, require_dependencies=False):
    manifest = json.loads((root / MANIFEST).read_text())
    for relative, entry in manifest["files"].items():
        check_entry(root, relative, entry)
    for relative, digest in manifest["code_sha256"].items():
        if sha(checked_file(root, relative)) != digest:
            raise ValueError(f"Runtime hash mismatch: {relative}")
    if require_dependencies:
        for relative, entry in manifest["dependencies"].items():
            check_entry(root, relative, entry)
    print("CHECKPOINT_BUNDLE_VERIFIED", manifest["update"], "dependencies", require_dependencies)


def import_inputs(root, source, evaluation_only=False):
    manifest = json.loads((root / MANIFEST).read_text())
    selected = {p: e for p, e in manifest["dependencies"].items()
                if not evaluation_only or e["role"] != "expert_training"}
    for relative, entry in selected.items():
        target = root / relative_path(relative)
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Destination escapes workspace: {relative}")
        if target.exists():
            check_entry(root, relative, entry)
            continue
        original = check_entry(source, relative, entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        with original.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        check_entry(root, relative, entry)
    print("CHECKPOINT_INPUTS_IMPORTED", len(selected), "evaluation_only", evaluation_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--require-dependencies", action="store_true")
    p = sub.add_parser("import-inputs")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--from-workspace", type=Path, required=True)
    p.add_argument("--evaluation-only", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        export(args.source.resolve(), args.out.resolve())
    elif args.command == "verify":
        verify(args.root.resolve(), args.require_dependencies)
    else:
        import_inputs(args.root.resolve(), args.from_workspace.resolve(), args.evaluation_only)
