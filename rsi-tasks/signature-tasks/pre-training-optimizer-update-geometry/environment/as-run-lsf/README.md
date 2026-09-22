# As-run LSF/Apptainer environment (archival)

These files are the sanitized environment that produced the recorded signature
runs in [`rsi-logs/signature-tasks/pre-training-optimizer-update-geometry/`](../../../../rsi-logs/signature-tasks/pre-training-optimizer-update-geometry/):
an Apptainer definition built on a prepared Megatron-Bridge runtime
(`optimizer_update_geometry.def`, `materialize_project.sh`), the project overlay
with the trusted pretraining entrypoint and candidate optimizer interface, and
the LSF/Slurm scaling-ladder tools (`task-tools/`, including the host-lease
helper). Private identifiers, site paths, and network addresses were removed;
the ladder, profiles, and scoring logic are unchanged.

They are kept for provenance only. The portable Harbor task in the parent
directory replaces them: its Dockerfile copies `runtime/`, `task-tools/`, and
`assets/` from the parent, never this directory, and Harness owns placement,
leases, and launch (see the task README). Do not run these scripts against the
portable image.
