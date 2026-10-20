# GARMI RL Workspace

A separate copy of the CRAM Workspace hosts the counter-to-table demo: CRAM navigation → RL base/right-arm approach → fixed close/lift → CRAM transport and placement.

The **GARMI RL** tab starts and stops runs, shows the live Isaac camera and actual task stages, and plays the recorded video. **NeemHub**, **VS Code**, and **JupyterLab** use the session’s existing proxy. Opening the page does not start a simulation.

For container and Binder launch instructions, see [binder/README.md](../../binder/README.md). The original CRAM Lab page remains unchanged.

Each run creates its own directory under `results/cram-rl-web/`. Success requires task completion and a passed physical placement check. Closing a run stops only that run’s processes.
