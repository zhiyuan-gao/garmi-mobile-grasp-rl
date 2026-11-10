"""Bind the published mug checkpoint and verified assets to the generic bridge."""

import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml
from garmi_cram_rl.contract import Bundle
from garmi_cram_rl.policy import Actor

CHECKPOINT_SHA = "b6ae2305fcbb353d88801ca3f9817b74665f34f9d8238b6fcd5a96d41e082ce2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_bundle(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mug = root / "household/mug"
    packaged = json.loads((root / "demos/household_sequence/assets.json").read_text())
    for name, expected in packaged["files"].items():
        if name.startswith("household/mug/") and sha(root / name) != expected:
            raise ValueError(f"Packaged mug resource changed: {name}")
    checkpoint = (
        mug / "runs/mug-ppo-physical-success/ppo-6500/checkpoints/update_6500.pt"
    )
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Unexpected mug policy checkpoint")
    profile = yaml.safe_load(
        (mug / "runs/mug-physical-success-review/ppo.yaml").read_text()
    )
    task = profile["garmi_task"]
    mask_path = root / "resources/validated-ycb-v6/mug/position-mask.json"
    if sha(mask_path) != task["kitchen"]["files_sha256"]["mask"]:
        raise ValueError("Original mug position mask changed")
    mask = json.loads(mask_path.read_text())
    for name, expected in mask["sources_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError(f"Original mask provenance changed: {name}")
    bank = json.loads((root / mask["bank_path"]).read_text())
    model = bank["objects"][0]
    for key in ("mesh", "usd"):
        path = mug / model[key].removeprefix("/workspace/")
        if sha(path) != model[key + "_sha256"]:
            raise ValueError(f"Mug {key} differs from training")
        model[key] = str(path)
    (output / "reference-bank.json").write_text(json.dumps(bank, indent=2))
    (output / "model.json").write_text(json.dumps(model, indent=2))
    # Only relocate resource paths. Keep all mask cells and grasp indices intact.
    original_mask = copy.deepcopy(mask)
    mask["sources_sha256"] = {
        str(root / name): expected for name, expected in mask["sources_sha256"].items()
    }
    mask["bank_path"] = str(output / "reference-bank.json")
    mask["sources_sha256"][mask["bank_path"]] = sha(output / "reference-bank.json")
    (output / "position-mask.json").write_text(json.dumps(mask, indent=2))
    task["multi_object"].pop("training_reset_pool", None)
    task["multi_object"]["manifest"] = str(output / "reference-bank.json")
    task["kitchen"]["mask"] = str(output / "position-mask.json")
    task["kitchen"]["scene_usd"] = str(root / "integration/assets/apartment.usda")
    task["kitchen"]["files_sha256"] = {
        key: sha(task["kitchen"][key]) for key in ("mask", "scene_usd")
    }
    (output / "profile.yaml").write_text(yaml.safe_dump(profile))
    actor = Actor(torch.load(checkpoint, map_location="cpu", weights_only=False)).eval()
    traced = torch.jit.trace(actor, torch.zeros(1, 64))
    traced.save(str(output / "actor.ts"))
    sample = torch.randn(8, 64)
    torch.testing.assert_close(actor(sample), traced(sample))
    manifest = dict(
        checkpoint_sha256=CHECKPOINT_SHA,
        action_contract="base_velocity_joint_delta_v1",
        bundle_files={
            name: sha(output / name)
            for name in (
                "reference-bank.json",
                "position-mask.json",
                "profile.yaml",
                "actor.ts",
                "model.json",
            )
        },
        workspace_files={
            **original_mask["sources_sha256"],
            str(mask_path): sha(mask_path),
            str(checkpoint): CHECKPOINT_SHA,
            **{model[key]: model[key + "_sha256"] for key in ("mesh", "usd")},
        },
        adaptation="Resource relocation and deterministic actor export; unchanged weights, references and mask cells",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return Bundle(output, root)
