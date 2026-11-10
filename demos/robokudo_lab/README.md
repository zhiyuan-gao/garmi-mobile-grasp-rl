# Local RoboKudo Lab preview

This opt-in Lab keeps the existing NeemHub + Desktop layout. The new **RoboKudo ↗**
entry opens a separate perception explorer. It displays the actual head RGB-D
snapshot, counter cloud, RoboKudo support plane, selected target cloud and custom
known-model alignment. The Three.js view supports orbit, zoom and overlay toggles.
CRAM's station and grasp selection are shown beside the perception measurements.

The task runs online in Isaac Sim. The original NeemHub scene, Plan, Statechart,
Knowledge, Kinematics and Transforms remain accessible through the existing arrow.
The perception explorer follows RoboKudo Webexplorer's node/output interaction;
it is our adapter, not a packaged copy of the official Webexplorer application.

## Runtime

Use a dedicated copy of the native deployment image and install the source-map
runtime. Set `GARMI_PERCEPTION_PREVIEW=1`, then execute:

```bash
/opt/conda/bin/python /workspace/demos/robokudo_lab/install_preview.py
```

Open the container's authenticated Jupyter Workspace at
`/files/demos/web_ui/index.html`. Connecting its Desktop starts the local control
service. Select **Mug** or **Bowl**, then **Start task**. Stop the retained final
simulation before selecting another object. Each start creates a new run folder.
No existing standalone or household task is overwritten in the source tree.

The installer is explicitly restricted to the preview environment. It adds the
Workspace link and a dedicated desktop autostart entry. In this disposable
container it also moves robot-state reads outside NeemHub's snapshot publication
lock, addressing the observed bridge/world-state lock inversion. The previous
source is saved beside it; hashes are recorded in `results/cram-rl-web/installation.json`.
No motion planner or policy is changed by this visualization fix.

## What the visualization means

- The displayed perception image is a **single initial snapshot**, not continuous
  visual tracking. It is labelled with its capture time.
- Plane/clustering outputs come from the actual RoboKudo worker. The model-fit
  node is explicitly labelled custom. Fixed orientation is stated in the UI.
- The exported point clouds are sampled for display only. Detection uses its
  original points. No ground-truth object pose is supplied to the explorer.
- Localization drives the existing CRAM and RL paths. Physical hold/contact
  verification remains simulation based.
- Task and decision status comes from real events, not an animated timer. The
  original CRAM plan is shown by NeemHub; the perception inspector shows the
  input/output evidence for this run.

The current preview performs one grasp and retains the final live scene. It does
not add perception to the complete bowl/table/mug/drawer sequence yet. The saved
bowl and mug experiment branches and the older Lab remain available separately.
