# Household Lab release preview

This candidate runs one continuous task in the full GARMI apartment: the bowl and
mug start on the kitchen counter, the bowl is delivered to the dining table, and
the mug is then picked up and placed inside the drawer before the left hand closes it.

The workspace embeds the original NeemHub interface alongside the live Isaac Sim
desktop. Plan opens by default. The bowl portion displays actual CRAM actions,
station checks, grasp selection and `RLPickUpAction`; the existing mug and drawer
controller publishes measured stage progress as monitor entries in the same list.
Start and Stop operate the simulation. Other recorded task entries remain available.

## Candidate versions

- Local branch in both repositories: `codex/lab-release-preview-20261106`.
- Packaged RL source: `9fe7eadba52c614ad962a9a2853e5c543bf64b6f`.
- Deployment source: `zhiyuan-gao/cram-vrb-lab`, local commit `6cdc429`.
- Image: `garmi-household:review-20261106`.
- Image ID: `sha256:da3eb3a3fe4bed3e0f4a852e37fb5b8558739e09cba71bbe63d1cdf041402874`.
- Environment: the pinned native Isaac Sim 6.1 base used by the household deployment.

The deployment source lives at `/home/zgao/FAME_RL/reviews/cram-lab-release-preview`.
It bundles the runtime built by `scripts/build_native_runtime.py`; local development
sources and training directories are not mounted into the review container.

## Local review

The review container is `garmi-household-preview-20261106`. Its launcher and private
Jupyter token are under `/home/zgao/FAME_RL/deployment/lab-release-preview-20261106`.
Run `open-preview.sh` there to open the authenticated workspace. The launcher binds
Jupyter to loopback port 8888 and explicitly uses `--runtime=nvidia --gpus all`.
The latter is necessary on this host to expose the Vulkan ICD as well as CUDA.

The same `demos/web_ui/index.html` and runtime are used locally and in deployment.
Jupyter supplies the local authentication or Binder user-session prefix. A writable
shader-cache mount speeds up subsequent local starts; run logs are saved separately.

## Validation on 2026-11-06

- 43 CRAM interface, skill selection, planning-copy and visualization tests passed
  against the packaged environment.
- All 209 runtime source-file hashes matched the release manifest.
- The complete physical task passed in one scene initialization, covering 160.8 s
  of simulation time. Bowl placement error was 5.9 mm. The drawer opened to 36.4 cm
  and finished within 0.5 mm of closed, with the mug verified inside.
- Browser inspection confirmed the five NeemHub tabs, real live scene meshes,
  the actual CRAM plan, measured mug/drawer stage updates and the Isaac desktop.
- The internal Statechart of the separate Giskard process is not yet published to
  NeemHub; that tab currently has no live motion-statechart graph. Plan contains the
  execution progress. Drawer control remains the existing IK implementation.

Evidence: `validation.json`, `tests.log`, `completed-preview.png` and `validated-run/`
in the local review directory. The visual-only changes applied during verification
were checked against the final packaged hashes before the final image was launched.

These commits and the image are local review candidates. GitHub branches and the
VRB Enter Lab link have not been updated by this preparation.
