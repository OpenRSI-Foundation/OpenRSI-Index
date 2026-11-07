# Set up Flow2 and Genkai

The packaged profiles provide reusable scheduler, storage and runtime settings
for new tasks on these sites. Set up the site runtime once, then complete a
profile copy with each task's image paths and external assets before validation.
Reuse a compatible existing installation. These steps install user-owned tools;
they do not change the site's Singularity installation or launch a research run.

| Profile | Site | Compute architecture | Allocation |
| --- | --- | --- | --- |
| `flow2` | Nagoya Flow 2.0 Type II, PBS Pro | aarch64 | `t2-share`: 1 GPU, 36 cores, 216 GiB RAM |
| `flow2-node` | Nagoya Flow 2.0 Type II, PBS Pro | aarch64 | `t2-single`: 4 GPUs, 144 cores, 864 GiB RAM |
| `genkai` | Kyushu Genkai group B, PJM | x86_64 | `b-batch`, `jobenv=singularity`: 30 cores and 226.7 GiB RAM per GPU |

These profiles check CPU/RAM requests against the selected allocation before
submission. Flow2 builds use the single-GPU share even for `flow2-node` runs;
Genkai builds also request one GPU. If you select a different build queue, set
its `scheduler.build.capacity` accordingly. CPU-only tasks can use a CPU queue
profile or a whole-node allocation; the GPU-share profiles require a GPU request.
Site references: [Flow2 queues](https://www.cc.nagoya-u.ac.jp/ja/usage/fees/job-classes/)
and [Genkai resource groups](https://www.cc.kyushu-u.ac.jp/scp/en/usage/resource_groups/).

## Install non-setuid Apptainer

The [upstream installer](https://github.com/apptainer/apptainer/blob/v1.5.5/tools/install-unprivileged.sh)
extracts Apptainer and its runtime dependencies into a user directory. It needs
`curl`, `rpm2cpio` and `cpio`. Select **one** site configuration below; the
directories match the packaged profiles. Use a fresh installation directory,
or reuse an existing installation that already satisfies this setup.

For Flow2, including `flow2-node`:

```bash
rsi_site_root="/data/$USER/rsi"
rsi_site_dist=el9
rsi_site_arch=aarch64
rsi_profile_name=flow2  # or flow2-node
rsi_scheduler=pbs
```

For Genkai:

```bash
rsi_site_root="$HOME/rsi"
rsi_site_dist=el8
rsi_site_arch=x86_64
rsi_profile_name=genkai
rsi_scheduler=pjm
```

Then download the pinned installer and run it as the cluster user, without sudo:

```bash
rsi_harness_root=/absolute/path/to/RSI-Harness
rsi_apptainer_root="$rsi_site_root/apptainer"
rsi_setup_dir=$(mktemp -d)
curl -fL --retry 3 \
  https://raw.githubusercontent.com/apptainer/apptainer/v1.5.5/tools/install-unprivileged.sh \
  -o "$rsi_setup_dir/install-unprivileged.sh"
bash "$rsi_setup_dir/install-unprivileged.sh" \
  -d "$rsi_site_dist" -a "$rsi_site_arch" -v 1.5.5 "$rsi_apptainer_root"
```

The profile's `apptainer.binary` should point to
`<rsi_apptainer_root>/bin/apptainer`, the installer's wrapper. Flow2 can download
the aarch64 packages from its login node, but must execute them on a compute
node. Genkai needs the additional patch below before GPU validation.

The profiles also enable rootless apt package installation and use the
installer's fakeroot helpers for builds without subordinate UID/GID ranges:
`<rsi_apptainer_root>/<architecture>/utils/bin/faked-sysv` and
`<rsi_apptainer_root>/<architecture>/utils/lib/libfakeroot/libfakeroot-sysv.so`.
When reusing another installation, check these two files and update
`builder.faked_binary` and `builder.fakeroot_library` to its compatible helpers.
They support image construction; they do not change the task's runtime user.

### Genkai: recursive device-directory binds

Genkai's job scope bind-mounts individual NVIDIA device nodes. Stock Apptainer
1.5.5 can fail with `mount /dev/nvidia-caps: invalid argument` because it binds
the containing directory non-recursively. The
[included patch](../scripts/cluster/apptainer-1.5.5-recursive-device-dir-bind.patch)
adds `MS_REC` for device directories; it does not expose devices outside the
job's device cgroup.

Build on a Genkai-compatible x86_64 host. This requires Go **1.26.10 or later**,
a C/C++ compiler, Make, `pkg-config`, `patch`, and libseccomp development
headers/libraries. The site's existing development tools or user modules can
supply them; do not disable seccomp to work around a missing dependency.

After the installer above, build the two patched binaries against the same
installation prefix. Retain its dependency helpers and wrapper scripts:

```bash
curl -fL --retry 3 \
  https://github.com/apptainer/apptainer/releases/download/v1.5.5/apptainer-1.5.5.tar.gz \
  -o "$rsi_setup_dir/apptainer-1.5.5.tar.gz"
tar -xzf "$rsi_setup_dir/apptainer-1.5.5.tar.gz" -C "$rsi_setup_dir"
cd "$rsi_setup_dir/apptainer-1.5.5"
patch -p1 < "$rsi_harness_root/scripts/cluster/apptainer-1.5.5-recursive-device-dir-bind.patch"
export GOFLAGS=-buildvcs=false
./mconfig -b builddir --without-suid \
  --prefix="$rsi_apptainer_root/x86_64" \
  --localstatedir="$rsi_apptainer_root/x86_64/var/lib"
make -C builddir -j8 "$(pwd -P)/builddir/apptainer" \
  "$(pwd -P)/builddir/cmd/starter/c/starter"

cp -p "$rsi_apptainer_root/x86_64/bin/apptainer" \
  "$rsi_apptainer_root/x86_64/bin/apptainer.rpm-1.5.5"
cp -p "$rsi_apptainer_root/x86_64/libexec/apptainer/libexec/starter" \
  "$rsi_apptainer_root/x86_64/libexec/apptainer/libexec/starter.rpm-1.5.5"
install -m 0755 builddir/apptainer "$rsi_apptainer_root/x86_64/bin/apptainer"
install -m 0755 builddir/cmd/starter/c/starter \
  "$rsi_apptainer_root/x86_64/libexec/apptainer/libexec/starter"
"$rsi_apptainer_root/bin/apptainer" --version
"$rsi_apptainer_root/bin/apptainer" buildcfg | grep -E 'SESSIONDIR|SUID'
```

The `libexec/apptainer/bin/starter` path is an installer wrapper: replace its
underlying `libexec/apptainer/libexec/starter` binary, as above. `buildcfg` should
report a non-setuid installation, and its `SESSIONDIR` must be an existing
directory of the installer's tree (`x86_64/var/lib/apptainer/mnt/session`):
`--localstatedir` keeps the installer's `var/lib` layout. This patch matches
the recorded Genkai 1.5.5 adaptation; another Apptainer version needs its own
compatibility check.

## Complete a task-specific profile

Keep the configured site profile outside the task repository. For each task,
derive a separate copy if its image or asset paths differ. To start a site copy:

```bash
mkdir -p "$rsi_site_root/profiles"
rsi_profile_path="$rsi_site_root/profiles/$rsi_profile_name.toml"
cp -n "$rsi_harness_root/src/rsi_harness/cluster/$rsi_scheduler/$rsi_profile_name.toml" \
  "$rsi_profile_path"
```

Inspect `task.toml`, the Dockerfile, its base image and any existing image
metadata to fill in these fields:

- `apptainer.workspace_target`: the effective image `WORKDIR`, including an
  inherited value. The template's `/workspace` is an example.
- `apptainer.container_python`: an actual Python 3.8+ executable in the task
  image. Resolve it inside the built image if the Dockerfile does not settle
  its absolute path; it is separate from the host Harness Python.
- `apptainer.work_binds` and `judge_binds`: actual shared host files required by
  the task's assets, with the declared container targets and phase access.
  Keep the lists empty for assets already in the image. Mount hidden Judge
  data only in Judge; do not add sample data paths just to fill a template.
- Shared storage, scheduler account/project and executable paths: use those
  assigned to the operator. If installation roots differ from the examples,
  update all corresponding profile paths.

The Harness supplies its pinned static Podman through `builder.podman_bundle`.
The task image must support the compute architecture: Flow2 is aarch64, so an
x86_64-only base image or wheel needs a task-environment compatibility repair.
Genkai's `resources.node_tmp_variable = "PJM_SSD_DIR"` selects each job's SSD
scratch automatically; do not replace it with a previous job's expanded path.

The original adaptations validated manually built and imported SIF images.
For a new task, validate the Dockerfile build as well as runtime execution;
reusing the site profile alone does not establish that a new image works.

## Controller and validation

On Flow2, the login nodes are x86_64 while GPU nodes are aarch64. Run the
controller in a compute job, using a separate aarch64 Harness environment and
native Agent CLIs at the versions pinned by that Harness. Keep the login
environment for offline checks and pass the compute executable to the skill
planner's `--controller-executable`. Both environments must use the same
Harness revision and shared task/profile paths. The controller job needs PBS
commands and Git access; budget it separately and give it enough walltime for
the run job's queue waits and execution. A controller job that ends first
cancels its run job; continue with `rsi-harness resume RUN_ID --cluster
PROFILE` from a new controller job. Do not use `qsub -V` to copy credentials
into the scheduler's job records. On Genkai, run the controller on a login node.
The PBS/PJM adapters submit separate build and run jobs. To queue once
instead, set `scheduler.launch = "allocation"` in a profile copy: the
controller job then runs the build and run jobs itself, so request the GPUs
the task plans and walltime for the build (unless its image is cached), the
run and five minutes more. Neither mode needs the original adaptation's
user-space Slurm or MariaDB; Genkai uses `pjsub`, `pjstat` and `pjdel`
directly.

Inside an ordinary compute allocation, run the cluster skill's site check with
the selected runtime and built task image:

```bash
bash /absolute/path/to/cluster-harbor-task-validator/scripts/site_check.sh \
  --apptainer "$rsi_apptainer_root/bin/apptainer" \
  --image /absolute/path/to/task.sif
```

Then run the skill preflight and Harness `--dry-run` against the completed
profile, followed by the validator's smoke and full baseline Judge. Dry-run
checks do not establish image or GPU execution acceptance. Preserve the
resolved profile and runtime versions with that validation evidence; the
runner reuses them for the experiment.
