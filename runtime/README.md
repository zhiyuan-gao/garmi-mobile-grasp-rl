# GARMI demo runtime

This branch packages the counter-to-table demo for a GPU-backed Binder session.
The original training entrypoints remain available.

`data.tar.gz` contains only the frozen inference policy, its grasp bank and position
mask, the adapted robot/bowl/kitchen assets, and the small GraspGen cache required
by the existing CRAM adapter. It contains no training trajectories or experiment runs.
`data-manifest.json` records every included file's SHA-256. The build checks these
hashes and the policy's original 37-file dependency manifest.

The complete apartment and the original CRAM stack come from the pinned upstream
Lab image. `integration/prepare_full_scene.py` constructs the complete demo scene
from those assets at build time. `runtime/cram_vrb_lab` and `giskard_server.py` are
the corresponding GARMI adapter files from yxzhan/cram-vrb-lab@41c1032, including
the fixes used in the local experiment. See `THIRD_PARTY_NOTICES.md` for upstream
sources; third-party asset and code terms continue to apply.

Launch path: `proxy/8781/`. JupyterHub supplies authentication and the session URL.
The page starts the simulator only when the user clicks the run button. Each run
saves a separate video and report under `results/cram-rl-web/`.
