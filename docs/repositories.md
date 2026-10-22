# Source and deployment repositories

## Source: garmi-mobile-grasp-rl

`main` is the common development branch for expert collection, imitation and PPO
training, model evaluation, the CRAM + RL bowl task, and the standalone mug drawer
task. The mug drawer implementation lives in `demos/mug_drawer/` and uses the
published update-6500 model followed by Cartesian IK for the two arms.

The native desktop runtime lives in `deployment/isaac61/`. Its source map retains
the exact Isaac Sim 6.1 deployment variants alongside the Isaac Sim 5.1 training
implementation. `scripts/build_native_runtime.py` exports a reproducible package
with the source commit and hashes.

The branches `codex/integrate-grasp-training-and-vrb`,
`codex/ycb-wrist-v6-ppo2000`, and `vrb-demo` retain integration and experiment
history. New source work starts from `main`. Published checkpoint tags continue
to identify their original code and model artifacts.

## Deployment: cram-vrb-lab

The fork at <https://github.com/zhiyuan-gao/cram-vrb-lab> maintains
`garmi-rl-native-demo` as the active deployment branch. It owns the Binder
Dockerfile, installation scripts, original workspace page and desktop setup.
`demos/garmi_rl/source.json` identifies the exact RL source used in its package.

The older `garmi-rl-demo` branch retains the earlier deployment approach.
The inherited `main`, `garmi`, `dev`, and `isaac6` branches preserve upstream
history. Deployment changes belong on `garmi-rl-native-demo`.

## Update flow

1. Implement and test task changes in the RL repository.
2. Commit the source and build the native package.
3. Update the package and provenance in the CRAM deployment repository.
4. Build and exercise the Lab on the production Binder service.
5. Set the VRB introduction page's Enter Lab link to the validated deployment SHA.

Production endpoints are `binder.intel4coro.de` and `jupyter.intel4coro.de`.
The `.dev` Binder endpoint is inaccessible in this deployment workflow.
The workspace route is `files/demos/web_ui/index.html`.

The VRB introduction page is managed separately through Redaxo. Repository
updates do not change its pinned Enter Lab link. Its current pin is
`cram-vrb-lab@9b7d8f3b21b350ca1dba986e57e377f416ee4935`.
