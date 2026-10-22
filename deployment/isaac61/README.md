# Native VRB runtime

This directory maintains the Isaac Sim 6.1 version of the bowl counter-to-table
demo used by the CRAM Lab desktop. It includes episode reset, continuous runs,
camera controls, the live viewer and the Isaac Lab compatibility patch.

`source-map.json` maps each packaged file to its versioned source. Shared files
come from the main project tree; `overrides/` contains deployment-specific files.
The initial import preserves all 146 file contents from the native Lab at
`zhiyuan-gao/cram-vrb-lab@9b7d8f3`. Training and the mug drawer demo use their
Isaac Sim 5.1 environments and frozen model configurations.

## Build the deployment package

From the repository root, after committing changes:

```bash
python3 scripts/build_native_runtime.py verify
python3 scripts/build_native_runtime.py build --out /path/to/new-runtime-package
```

This CPU-only command produces `runtime.tar.gz`, `files.sha256.json` and
`source.json`. The latter pins the source commit and both artifact hashes.
Copy these three files to `demos/garmi_rl/` on the deployment repository's
`garmi-rl-native-demo` branch. Its image build installs and verifies the package.

When editing a runtime file, use its source path in `source-map.json`, update its
SHA256 in that map, and run the relevant tests before building. Every packaged
source must be tracked by Git. Unchanged source and file modes produce the same
archive bytes across builds.

The mug-to-drawer task is maintained at `demos/mug_drawer/`. Its documented
launcher uses the mug update-6500 runtime and Apptainer. The native package here
provides the bowl desktop task; adding a mug desktop entry is a deployment step
with its own simulator and asset validation.
