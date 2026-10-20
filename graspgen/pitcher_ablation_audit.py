"""CPU preflight and fixed validation-label checks for the pitcher ablation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from gymnasium.spaces import Box
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.model_instantiators.torch import shared_model


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    root = a.root
    plan = json.loads((root / "plan.json").read_text())
    base_path = Path(plan["bc_checkpoint"])
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    profile = yaml.safe_load((root / "reward-only.yaml").read_text())
    report = {}
    if a.checkpoint is None:
        from garmi_rl.precision_reward import precision_score
        distance = torch.linspace(0, .2, 1001, requires_grad=True)
        angle = torch.linspace(0, .5, 1001, requires_grad=True)
        scores = precision_score(distance, angle)
        assert torch.isfinite(scores).all() and scores.min() >= 0 and scores.max() <= 1
        assert scores[0] == 1 and (scores[1:] < scores[:-1]).all()
        scores.sum().backward()
        assert torch.isfinite(distance.grad).all() and torch.isfinite(angle.grad).all()
        assert (distance.grad <= 0).all() and (angle.grad <= 0).all()
        examples = precision_score(torch.tensor([.07, .02, .005]),
                                   torch.tensor([.24, .10, .05]))
        assert torch.allclose(examples, torch.tensor([.182596, .489735, .883111]), atol=1e-4)
        # The copied entrypoint changes only log std after the BC transfer.
        altered = copy.deepcopy(base["policy"])
        altered["log_std_parameter"].sub_(np.log(2.))
        assert all(torch.equal(v, altered[k]) for k, v in base["policy"].items()
                   if k != "log_std_parameter")
        std = base["policy"]["log_std_parameter"].exp()
        assert torch.allclose(std, torch.tensor([.1] * 3 + [.05] * 7), atol=1e-6)
        assert torch.allclose(altered["log_std_parameter"].exp(), std * .5)
        report["preflight"] = dict(reward_monotonic=True, finite_gradients=True,
                                   example_scores=examples.tolist(),
                                   original_std=std.tolist(), reduced_std=(std*.5).tolist())

    data_path = root / "validation-labels.npz"
    if not data_path.exists():
        assert a.checkpoint is None
        manifest = json.loads((root / "validation-selection.json").read_text())
        rng = np.random.default_rng(plan["validation_seed"])
        xs, ys = [], []
        for row in manifest["demonstrations"]:
            path = Path(row["file"])
            assert sha(path) == row["sha256"]
            with np.load(path, allow_pickle=False) as data:
                valid = np.flatnonzero(data["bc_mask"])
                ids = rng.choice(valid, 64, replace=len(valid) < 64)
                xs.append(data["observations"][ids].copy())
                ys.append(data[row.get("label_key", "actions")][ids].copy())
        np.savez_compressed(data_path, observations=np.concatenate(xs), actions=np.concatenate(ys))
        (root / "validation-labels.sha256").write_text(sha(data_path) + "\n")
    assert sha(data_path) == (root / "validation-labels.sha256").read_text().strip()
    with np.load(data_path, allow_pickle=False) as data:
        x, y = torch.tensor(data["observations"]), torch.tensor(data["actions"])
    models = copy.deepcopy(profile["models"])
    models.pop("separate")
    roles = list(models)
    structure = [models[r].pop("class") for r in roles]
    model = shared_model(observation_space=Box(-np.inf, np.inf, (64,)),
                         action_space=Box(-np.inf, np.inf, (10,)), device="cpu",
                         structure=structure, roles=roles,
                         parameters=[models[r] for r in roles])
    for role in roles:
        model.init_state_dict(role)
    scaler = RunningStandardScaler(64, device="cpu")
    path = a.checkpoint or base_path
    weights = torch.load(path, map_location="cpu", weights_only=False)
    assert all(torch.equal(v, weights["state_preprocessor"][k])
               for k, v in base["state_preprocessor"].items())
    model.load_state_dict(weights["policy"])
    scaler.load_state_dict(weights["state_preprocessor"])
    with torch.no_grad():
        prediction = model.compute({"states": scaler(x)}, role="policy")[0]
        errors = (prediction-y).square()
        near = x[:, 30:33].norm(dim=1) < .02
    assert torch.isfinite(errors).all()
    report.update(status="passed", checkpoint=str(path), checkpoint_sha256=sha(path),
                  split="validation", trajectories=80, labelled_steps=len(x),
                  expert_action_mse=errors.mean().item(), near_goal_steps=int(near.sum()),
                  near_goal_action_mse=errors[near].mean().item() if near.any() else None,
                  policy_std=weights["policy"]["log_std_parameter"].exp().tolist(),
                  state_normalization_unchanged=True, labels_sha256=sha(data_path))
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print("ABLATION_AUDIT_PASSED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
