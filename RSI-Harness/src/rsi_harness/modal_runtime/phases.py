"""Private namespace phases within one Modal allocation, without Docker."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.judge_agent.judge_isolation import seccomp_program
from rsi_harness.models import AgentRunResult, ContainerRef, WorkQuiescence
from rsi_harness.runtime.output import BoundedBytes
from rsi_harness.runtime.redaction import redact_exact_values, redact_text

WORK_UID = 60000
SYSTEM_PATHS = ("/usr", "/bin", "/lib", "/lib64", "/etc")


def copy_regular_tree(source: Path, destination: Path) -> str:
    """Copy bounded regular data, never follow a candidate link or device."""
    total = count = 0
    digest = hashlib.sha256()
    destination.mkdir(parents=True, exist_ok=False)
    for parent, directories, files in os.walk(source, followlinks=False):
        parent = Path(parent)
        directories.sort()
        for name in sorted(directories + files):
            path = parent / name
            rel = path.relative_to(source)
            info = path.lstat()
            count += 1
            if count > 20000:
                raise InfrastructureError("candidate exceeds snapshot entry limit")
            if stat.S_ISDIR(info.st_mode):
                (destination / rel).mkdir(mode=0o755)
                digest.update(json.dumps([str(rel), "directory", 0o755]).encode())
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise InfrastructureError(
                    f"snapshot refuses non-regular or linked artifact: {rel}"
                )
            total += info.st_size
            if total > 512 * 1024 * 1024:
                raise InfrastructureError("candidate exceeds snapshot byte/file limit")
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                content = stream.read(info.st_size + 1)
            if len(content) != info.st_size:
                raise InfrastructureError("candidate changed during snapshot")
            target = destination / rel
            target.write_bytes(content)
            mode = 0o755 if info.st_mode & 0o111 else 0o644
            target.chmod(mode)
            digest.update(
                json.dumps(
                    [str(rel), mode, hashlib.sha256(content).hexdigest()]
                ).encode()
            )
    return digest.hexdigest()


def uid_processes(uid: int) -> dict[int, str]:
    found = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = dict(
                line.split(":", 1)
                for line in (p / "status").read_text().splitlines()
                if ":" in line
            )
            if int(fields["Uid"].split()[1]) == uid:
                found[int(p.name)] = fields["State"].strip()[0]
        except (FileNotFoundError, ProcessLookupError):
            pass
    return found


class NamespaceRuntime:
    def __init__(self, root, plan, config, run_id):
        self.root = Path(root)
        self.plan, self.config, self.run_id = plan, config, run_id
        self.workspace = self.root / "workspace"
        self.home = self.root / "home"
        self.work_tmp = self.root / "work-tmp"
        self.copies = self.root / "copies"
        self.codex_etc = self.root / "codex-etc"
        self.rounds = self.root / "rounds"
        self.feedback = self.network = self.process = None
        self.bindings = {}
        self.paused = False
        self.lock = threading.RLock()
        self.work_ref = ContainerRef(container_id=f"modal-{run_id}-work", role="work")

    def initialize(self, resume=False):
        if os.geteuid() != 0:
            raise SetupError("Modal controller requires root in its private allocation")
        if uid_processes(WORK_UID):
            raise SetupError("reserved Work uid is already in use")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        if not resume:
            seed = Path("/workspace")
            if seed.is_dir():
                copy_regular_tree(seed, self.workspace)
            else:
                self.workspace.mkdir()
            for p in (
                self.home,
                self.work_tmp,
                self.copies,
                self.rounds,
                self.codex_etc,
            ):
                p.mkdir()
        elif not all(
            p.is_dir() for p in (self.workspace, self.home, self.work_tmp, self.rounds)
        ):
            raise SetupError("retained Work state is incomplete")
        for base in (self.workspace, self.home, self.work_tmp, self.codex_etc):
            for parent, dirs, files in os.walk(base):
                os.chown(parent, WORK_UID, WORK_UID)
                for name in dirs + files:
                    os.chown(
                        Path(parent) / name, WORK_UID, WORK_UID, follow_symlinks=False
                    )
        for p in Path("/dev").glob("nvidia*"):
            for node in list(p.rglob("*")) if p.is_dir() else [p]:
                if stat.S_ISCHR(node.lstat().st_mode):
                    node.chmod(0o600)
        (self.root / "trusted-tools").mkdir(exist_ok=True)
        tests = self.root / "tests"
        if not tests.exists():
            copy_regular_tree(self.plan.task.source_dir / "tests", tests)
            tests.chmod(0o700)

    def copy_to(self, container, source, target):
        self._require_work(container)
        target = PurePosixPath(target)
        if not any(
            target.is_relative_to(p)
            for p in (PurePosixPath("/tmp"), PurePosixPath("/etc/codex"))
        ):
            raise SetupError(f"unsupported Agent hook destination: {target}")
        out = self.copies / hashlib.sha256(str(target).encode()).hexdigest()
        shutil.copyfile(source, out)
        out.chmod(source.stat().st_mode & 0o777)
        self.bindings[str(target)] = out

    def command(
        self,
        argv,
        *,
        judge=False,
        snapshot=None,
        logs=None,
        trajectory=None,
        environment=None,
        seccomp_fd=None,
    ):
        # gVisor provides loopback in unshare's private netns. bwrap's own
        # loopback RTM_NEWADDR configuration is unsupported there.
        cmd = [
            "unshare",
            "--user",
            "--map-users=0,0,65536",
            "--map-groups=0,0,65536",
            "--net",
            "bwrap",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
        ]
        for path in SYSTEM_PATHS:
            if Path(path).exists():
                cmd += ["--ro-bind", path, path]
        cmd += [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/run",
            "--tmpfs",
            "/home",
            "--tmpfs",
            "/root",
            "--tmpfs",
            "/var",
            "--dir",
            "/opt",
        ]
        cmd += [
            "--ro-bind" if judge else "--bind",
            str(snapshot if judge else self.workspace),
            "/workspace",
        ]
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/root" if judge else "/home/agent",
            "LANG": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
        }
        env.update(environment or {})
        if judge:
            cmd += [
                "--ro-bind",
                str(self.root / "tests"),
                "/tests",
                "--bind",
                str(logs),
                "/logs/verifier",
            ]
            if trajectory:
                cmd += ["--ro-bind", str(trajectory), "/logs/agent"]
            from rsi_harness.judge_agent import (
                JUDGE_ISOLATE_CODE_TARGET,
                install_judge_isolate,
            )

            code = install_judge_isolate(self.root / "trusted-tools")
            cmd += [
                "--ro-bind",
                str(code),
                str(JUDGE_ISOLATE_CODE_TARGET),
                "--ro-bind",
                str(code / "rsi-isolate"),
                "/usr/local/bin/rsi-isolate",
            ]
            if self.plan.task.verifier.gpu_count:
                cmd += ["--ro-bind", "/sys", "/sys"]
                for path in sorted(Path("/dev").glob("nvidia*")):
                    cmd += ["--dev-bind", str(path), str(path)]
            cmd += ["--cap-drop", "ALL"]
            for cap in (
                "CAP_SETUID",
                "CAP_SETGID",
                "CAP_CHOWN",
                "CAP_DAC_OVERRIDE",
                "CAP_FOWNER",
                "CAP_KILL",
            ):
                cmd += ["--cap-add", cap]
            env.update(
                RSI_JUDGE_ISOLATION_FALLBACK="none",
                RSI_JUDGE_ALLOW_MISSING_LANDLOCK_SIGNAL="0",
                RSI_JUDGE_NETNS="private",
            )
        else:
            cmd += [
                "--bind",
                str(self.home),
                "/home/agent",
                "--bind",
                str(self.work_tmp),
                "/tmp",
                "--bind",
                str(self.codex_etc),
                "/etc/codex",
            ]
            cmd += ["--perms", "0755", "--dir", "/run/rsi-harness"]
            if self.feedback:
                cmd += ["--ro-bind", str(self.feedback), "/run/rsi-harness/feedback"]
            if self.network:
                cmd += ["--ro-bind", str(self.network.root), "/run/rsi-harness/network"]
                env["RSI_SUBMIT_SOCKET"] = "/run/rsi-harness/network/submit.sock"
            for target, source in sorted(self.bindings.items()):
                cmd += ["--ro-bind", str(source), target]
            if self.plan.task.agent.name == "oracle":
                cmd += ["--bind", str(self.root / "solution"), "/solution"]
            # bwrap's --uid requires it to create the user namespace itself.
            # Our full uid-map is established by unshare; drop identity inside
            # it before executing any Agent code, with no_new_privs enforced.
            cmd += [
                "--cap-drop",
                "ALL",
                "--cap-add",
                "CAP_SETUID",
                "--cap-add",
                "CAP_SETGID",
            ]
            argv = [
                "setpriv",
                f"--reuid={WORK_UID}",
                f"--regid={WORK_UID}",
                "--clear-groups",
                "--no-new-privs",
                "--",
                *argv,
            ]
        if judge:
            cmd += ["--perms", "0700", "--dir", "/run/rsi-private-assets"]
        for index, asset in enumerate(self.config.assets):
            if asset.phase == ("judge" if judge else "work"):
                if judge:
                    # Gate private assets even on kernels without Landlock.
                    # Their volume-owned permissions cannot be changed in place.
                    private = f"/run/rsi-private-assets/{index}"
                    cmd += [
                        "--ro-bind",
                        str(asset.source),
                        private,
                        "--symlink",
                        private,
                        str(asset.target),
                    ]
                else:
                    cmd += ["--ro-bind", str(asset.source), str(asset.target)]
        if seccomp_fd is not None:
            cmd += ["--seccomp", str(seccomp_fd)]
        for key, value in env.items():
            cmd += ["--setenv", key, str(value)]
        return cmd + ["--chdir", "/workspace", "--", *argv]

    def run(
        self,
        argv,
        *,
        judge=False,
        snapshot=None,
        logs=None,
        trajectory=None,
        timeout=60,
        environment=None,
        output_path=None,
        secrets=(),
        on_exec_start=None,
    ):
        if timeout is None or timeout <= 0:
            raise SetupError("phase execution needs a finite positive timeout")
        fd = os.memfd_create("rsi-phase-filter")
        try:
            os.write(
                fd,
                seccomp_program(
                    no_network=False,
                    truncate_guard=False,
                    trace_guard=True,
                    namespace_guard=True,
                ),
            )
            os.lseek(fd, 0, 0)
            cmd = self.command(
                argv,
                judge=judge,
                snapshot=snapshot,
                logs=logs,
                trajectory=trajectory,
                environment=environment,
                seccomp_fd=fd,
            )
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                pass_fds=(fd,),
                start_new_session=True,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"},
            )
        finally:
            os.close(fd)
        if not judge:
            with self.lock:
                self.process = p
        if on_exec_start:
            on_exec_start(time.monotonic())
        captured = BoundedBytes(1_000_000)
        output = open(output_path, "ab") if output_path else None
        read_error = []

        def drain():
            try:
                # Bound hostile lines; never emit a partial credential-bearing line.
                while line := p.stdout.readline(1024 * 1024 + 1):
                    if len(line) > 1024 * 1024:
                        read_error.append(InfrastructureError("oversized output line"))
                        while line and not line.endswith(b"\n"):
                            line = p.stdout.readline(1024 * 1024)
                        line = b"[oversized output line omitted]\n"
                    safe = redact_text(
                        redact_exact_values(line.decode(errors="replace"), secrets)
                    ).encode()
                    captured.append(safe)
                    if output:
                        output.write(safe)
                        output.flush()
            except BaseException as error:
                read_error.append(error)

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        expired = False
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            expired = True
            self._kill(p)
        finally:
            if p.poll() is None:
                self._kill(p)
            reader.join(timeout=10)
            if output:
                output.close()
            if not judge:
                with self.lock:
                    if self.process is p:
                        self.process = None
            if reader.is_alive() or read_error:
                raise InfrastructureError("phase output could not be fully captured")
        return AgentRunResult(
            exit_code=p.returncode,
            output=captured.value.decode(errors="replace"),
            timed_out=expired,
            exec_started=True,
            full_output_captured=output_path is not None,
            output_truncated=captured.truncated,
        )

    def exec(
        self,
        container,
        command,
        *,
        timeout_seconds=None,
        user=None,
        environment=None,
        output_path=None,
        output_redact_values=(),
        output_callback=None,
        on_exec_start=None,
    ):
        self._require_work(container)
        argv = (
            ("/bin/sh", "-c", command) if isinstance(command, str) else tuple(command)
        )
        if self.network and self.network.provider_socket:
            argv = self.network.wrap(
                argv, relay_path=PurePosixPath("/usr/local/bin/rsi-netns-relay")
            )
        return self.run(
            argv,
            timeout=timeout_seconds or 60,
            environment=environment,
            output_path=output_path,
            secrets=output_redact_values,
            on_exec_start=on_exec_start,
        )

    def pause(self, container):
        self._require_work(container)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for pid in uid_processes(WORK_UID):
                try:
                    os.kill(pid, signal.SIGSTOP)
                except ProcessLookupError:
                    pass
            current = uid_processes(WORK_UID)
            if all(state in {"T", "t", "Z"} for state in current.values()):
                self.paused = True
                return
            time.sleep(0.01)
        raise InfrastructureError("Work did not reach an attested stopped state")

    def unpause(self, container):
        self._require_work(container)
        for pid in uid_processes(WORK_UID):
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        self.paused = False

    def stop(self, container):
        self._require_work(container)
        with self.lock:
            if self.process:
                self._kill(self.process)
        for pid in uid_processes(WORK_UID):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    @staticmethod
    def _kill(p):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait(timeout=10)

    def quiescence(self, container):
        if not uid_processes(WORK_UID):
            return WorkQuiescence.STOPPED
        self.pause(container)
        return WorkQuiescence.PAUSED

    def _require_work(self, container):
        if container != self.work_ref:
            raise InfrastructureError("phase reference differs from this run")
