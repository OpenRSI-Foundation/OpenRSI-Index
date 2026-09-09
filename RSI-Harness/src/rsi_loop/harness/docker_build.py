# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Docker image building for RSI Loop, adapted from SWE-bench's build_image pattern."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import docker
import docker.errors

from rsi_loop.harness.config import RSILoopConfig, get_env_directives, get_build_args, get_build_secrets
from rsi_loop.harness.constants import (
    BASE_IMAGE_BUILD_DIR,
    WORK_IMAGE_BUILD_DIR,
    JUDGE_IMAGE_BUILD_DIR,
    UTF8,
)
from rsi_loop.harness.dockerfiles import (
    get_dockerfile_base,
    get_dockerfile_work,
    get_dockerfile_judge,
)
from rsi_loop.harness.task_spec import TaskSpec


_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

_base_image_locks: dict[str, threading.Lock] = {}
_base_image_locks_guard = threading.Lock()
_base_image_rebuilt: set[str] = set()

_base_pull_locks: dict[str, threading.Lock] = {}
_base_pull_locks_guard = threading.Lock()

_base_push_locks: dict[str, threading.Lock] = {}
_base_push_locks_guard = threading.Lock()
_pushed_base_refs: set[str] = set()


def ansi_escape(s: str) -> str:
    return _ANSI_ESCAPE.sub("", s)


class BuildImageError(Exception):
    def __init__(self, image_name: str, message: str, logger: logging.Logger):
        super().__init__(message)
        self.image_name = image_name
        self.log_path = getattr(logger, "log_file", None)

    def __str__(self):
        return (
            f"Error building image {self.image_name}: {super().__str__()}\n"
            f"Check ({self.log_path}) for more information."
        )


def setup_logger(
    name: str, log_file: Path, mode: str = "w", verbose: bool = False
) -> logging.Logger:
    """Create a file logger for build processes.

    When *verbose* is True an additional StreamHandler on stdout is attached
    so that log messages are printed to the terminal in real time.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"rsi_loop.{name}.{log_file.name}")
    handler = logging.FileHandler(log_file, mode=mode, encoding=UTF8)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if verbose:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, "log_file", log_file)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def build_image(
    image_name: str,
    setup_scripts: dict[str, str],
    dockerfile: str,
    platform: str,
    client: docker.DockerClient,
    build_dir: Path,
    nocache: bool = False,
    buildargs: dict[str, str] | None = None,
    extra_hosts: dict[str, str] | None = None,
    verbose: bool = False,
    secrets: dict[str, str] | None = None,
) -> None:
    """
    Build a Docker image. Writes scripts + Dockerfile to build_dir, then calls docker build.
    When *secrets* is provided, uses ``docker build`` CLI with BuildKit
    ``--secret`` flags so that secret values never enter any image layer.
    """
    logger = setup_logger(image_name, build_dir / "build_image.log", verbose=verbose)
    logger.info(
        f"Building image {image_name}\n"
        f"Dockerfile:\n{dockerfile}\n"
        f"Setup scripts: {list(setup_scripts.keys())}"
    )

    try:
        # Write setup scripts
        for script_name, script_content in setup_scripts.items():
            script_path = build_dir / script_name
            with open(script_path, "w") as f:
                f.write(script_content)
            logger.info(f"[SETUP SCRIPT] {script_name}:\n{script_content}")

        # Write Dockerfile
        dockerfile_path = build_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        # Build
        logger.info(f"Building {image_name} in {build_dir} (platform={platform})")

        if secrets:
            _build_with_secrets(
                image_name=image_name,
                build_dir=build_dir,
                platform=platform,
                secrets=secrets,
                buildargs=buildargs,
                extra_hosts=extra_hosts,
                nocache=nocache,
                logger=logger,
            )
        else:
            response = client.api.build(
                path=str(build_dir),
                tag=image_name,
                rm=True,
                forcerm=True,
                decode=True,
                platform=platform,
                nocache=nocache,
                buildargs=buildargs or {},
                extra_hosts=extra_hosts,
            )

            buildlog = ""
            for chunk in response:
                if "stream" in chunk:
                    chunk_stream = ansi_escape(chunk["stream"])
                    logger.info(chunk_stream.strip())
                    buildlog += chunk_stream
                elif "errorDetail" in chunk:
                    logger.error(f"Error: {ansi_escape(chunk['errorDetail']['message'])}")
                    raise docker.errors.BuildError(
                        chunk["errorDetail"]["message"], buildlog
                    )

        logger.info("Image built successfully!")

    except docker.errors.BuildError as e:
        logger.error(f"BuildError during {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    except Exception as e:
        logger.error(f"Error during {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    finally:
        close_logger(logger)


def _build_with_secrets(
    image_name: str,
    build_dir: Path,
    platform: str,
    secrets: dict[str, str],
    buildargs: dict[str, str] | None = None,
    extra_hosts: dict[str, str] | None = None,
    nocache: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Build via ``docker build`` CLI with BuildKit ``--secret`` flags."""
    secret_files: list[Path] = []
    try:
        cmd = ["docker", "build", "--tag", image_name, "--platform", platform,
               "--rm", "--force-rm"]

        for sid, value in secrets.items():
            secret_path = build_dir / f".secret_{sid}"
            secret_path.write_text(value)
            secret_files.append(secret_path)
            cmd += ["--secret", f"id={sid},src={secret_path}"]

        for k, v in (buildargs or {}).items():
            cmd += ["--build-arg", f"{k}={v}"]

        if extra_hosts:
            for host, ip in extra_hosts.items():
                cmd += ["--add-host", f"{host}:{ip}"]

        if nocache:
            cmd.append("--no-cache")

        cmd.append(str(build_dir))

        env = {**os.environ, "DOCKER_BUILDKIT": "1"}
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        buildlog = ""
        for line in proc.stdout:
            stripped = ansi_escape(line.rstrip())
            if logger:
                logger.info(stripped)
            buildlog += line

        proc.wait()

        if proc.returncode != 0:
            raise docker.errors.BuildError(buildlog or "unknown error", buildlog)

    finally:
        for p in secret_files:
            p.unlink(missing_ok=True)


def _image_exists(client: docker.DockerClient, image_name: str) -> bool:
    try:
        client.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        return False


def _pull_remote_image(
    client: docker.DockerClient,
    remote_ref: str,
    local_tag: str,
) -> bool:
    """Try to pull a remote image and retag it locally. Returns True on success."""
    try:
        print(f"[registry] Pulling {remote_ref} ...")
        client.images.pull(remote_ref)
        image = client.images.get(remote_ref)
        image.tag(local_tag)
        print(f"[registry] OK → retagged as {local_tag}")
        return True
    except (docker.errors.NotFound, docker.errors.APIError) as e:
        print(f"[registry] Pull failed: {e}")
        return False


def _push_image(
    client: docker.DockerClient,
    local_tag: str,
    remote_ref: str,
) -> bool:
    """Tag a local image with the remote ref and push it. Returns True on success."""
    try:
        image = client.images.get(local_tag)
        image.tag(remote_ref)
        print(f"[registry] Pushing {remote_ref} ...")
        for chunk in client.images.push(remote_ref, stream=True, decode=True):
            if "error" in chunk:
                print(f"[registry] Push error: {chunk['error']}")
                return False
        print(f"[registry] Pushed {remote_ref}")
        return True
    except (docker.errors.ImageNotFound, docker.errors.APIError) as e:
        print(f"[registry] Push failed: {e}")
        return False


def build_base_image(
    task_spec: TaskSpec,
    config: RSILoopConfig,
    client: docker.DockerClient,
    force_rebuild: bool = False,
    verbose: bool = False,
) -> str:
    """Build the base image for a task. Returns the image tag.

    A per-key lock ensures that concurrent builds for the same base image
    serialize rather than racing to build the same image twice.
    """
    base_key = task_spec.base_image
    image_name = task_spec.base_image_tag

    with _base_image_locks_guard:
        if base_key not in _base_image_locks:
            _base_image_locks[base_key] = threading.Lock()
        lock = _base_image_locks[base_key]

    with lock:
        already_rebuilt = base_key in _base_image_rebuilt
        if already_rebuilt and _image_exists(client, image_name):
            return image_name
        if not force_rebuild and _image_exists(client, image_name):
            return image_name

        build_dir = BASE_IMAGE_BUILD_DIR / image_name.replace(":", "__")
        build_dir.mkdir(parents=True, exist_ok=True)

        env_directives = get_env_directives(config)
        dockerfile = get_dockerfile_base(
            task_spec.base_image_spec, task_spec.platform, env_directives,
            apt_mirror_url=config.apt_mirror_url,
        )

        build_image(
            image_name=image_name,
            setup_scripts={},
            dockerfile=dockerfile,
            platform=task_spec.platform,
            client=client,
            build_dir=build_dir,
            nocache=force_rebuild,
            buildargs=get_build_args(config),
            extra_hosts=config.extra_hosts,
            verbose=verbose,
        )

        _base_image_rebuilt.add(base_key)
        return image_name


def build_work_image(
    task_spec: TaskSpec,
    config: RSILoopConfig,
    client: docker.DockerClient,
    force_rebuild: bool = False,
    verbose: bool = False,
) -> str:
    """Build the work image (agent workspace). Returns the image tag."""
    image_name = task_spec.work_image_key

    if not force_rebuild and _image_exists(client, image_name):
        return image_name

    if not task_spec.work_needs_build:
        raise BuildImageError(
            image_name,
            f"Pre-built image '{image_name}' not found locally. "
            f"Pull it from the registry first (rsi-loop pull --task {task_spec.task_id}).",
            logging.getLogger(__name__),
        )

    # Ensure base image exists
    build_base_image(
        task_spec, config, client, force_rebuild=False,
        verbose=verbose,
    )

    build_dir = WORK_IMAGE_BUILD_DIR / image_name.replace(":", "__")
    build_dir.mkdir(parents=True, exist_ok=True)

    env_directives = get_env_directives(config)
    secrets = get_build_secrets(config)
    dockerfile = get_dockerfile_work(
        platform=task_spec.platform,
        base_image=task_spec.base_image_tag,
        cwd=task_spec.cwd,
        env_directives=env_directives,
        secrets=secrets or None,
    )

    build_image(
        image_name=image_name,
        setup_scripts={"setup_workspace.sh": task_spec.setup_workspace_script},
        dockerfile=dockerfile,
        platform=task_spec.platform,
        client=client,
        build_dir=build_dir,
        nocache=force_rebuild,
        buildargs=get_build_args(config),
        extra_hosts=config.extra_hosts,
        verbose=verbose,
        secrets=secrets or None,
    )

    return image_name


def build_judge_image(
    task_spec: TaskSpec,
    config: RSILoopConfig,
    client: docker.DockerClient,
    force_rebuild: bool = False,
    verbose: bool = False,
) -> str:
    """Build the judge image (grading environment). Returns the image tag."""
    image_name = task_spec.judge_image_key

    if not force_rebuild and _image_exists(client, image_name):
        return image_name

    if not task_spec.judge_needs_build:
        raise BuildImageError(
            image_name,
            f"Pre-built image '{image_name}' not found locally. "
            f"Pull it from the registry first (rsi-loop pull --task {task_spec.task_id}).",
            logging.getLogger(__name__),
        )

    # Ensure base image exists
    build_base_image(
        task_spec, config, client, force_rebuild=False,
        verbose=verbose,
    )

    build_dir = JUDGE_IMAGE_BUILD_DIR / image_name.replace(":", "__")
    build_dir.mkdir(parents=True, exist_ok=True)

    env_directives = get_env_directives(config)
    secrets = get_build_secrets(config)
    dockerfile = get_dockerfile_judge(
        platform=task_spec.platform,
        base_image=task_spec.base_image_tag,
        cwd=task_spec.cwd,
        env_directives=env_directives,
        secrets=secrets or None,
    )

    build_image(
        image_name=image_name,
        setup_scripts={
            "setup_judge.sh": task_spec.setup_judge_script,
        },
        dockerfile=dockerfile,
        platform=task_spec.platform,
        client=client,
        build_dir=build_dir,
        nocache=force_rebuild,
        buildargs=get_build_args(config),
        extra_hosts=config.extra_hosts,
        verbose=verbose,
        secrets=secrets or None,
    )

    return image_name


def build_all_images(
    task_spec: TaskSpec,
    config: RSILoopConfig,
    client: docker.DockerClient,
    force_rebuild: bool = False,
    force_rebuild_base: bool = False,
    verbose: bool = False,
) -> tuple[str, str, str]:
    """Build base + work + judge images for a task. Returns (base, work, judge) image tags."""
    base_name = task_spec.base_image_tag
    work_name = task_spec.work_image_key
    judge_name = task_spec.judge_image_key

    print(f"[build] Stage 1/2: base image  ({base_name})")
    base = build_base_image(
        task_spec, config, client, force_rebuild_base,
        verbose=verbose,
    )

    print(f"[build] Stage 2/2: work + judge images")
    print(f"         work:  {work_name}")
    print(f"         judge: {judge_name}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        work_fut = ex.submit(
            build_work_image, task_spec, config, client, force_rebuild, verbose=verbose,
        )
        judge_fut = ex.submit(
            build_judge_image, task_spec, config, client, force_rebuild, verbose=verbose,
        )
        work = work_fut.result()
        judge = judge_fut.result()

    return base, work, judge


def pull_all_images(
    task_spec: TaskSpec,
    registry: str,
    client: docker.DockerClient,
) -> tuple[bool, bool, bool]:
    """Pull base + work + judge images from registry. Returns (base_ok, work_ok, judge_ok).

    A per-key lock prevents redundant base image pulls when multiple
    tasks sharing the same base image are pulled in parallel.
    """
    base_key = task_spec.base_image
    with _base_pull_locks_guard:
        if base_key not in _base_pull_locks:
            _base_pull_locks[base_key] = threading.Lock()
        lock = _base_pull_locks[base_key]

    with lock:
        local_tag = task_spec.base_image_tag
        if _image_exists(client, local_tag):
            base_ok = True
        else:
            base_ok = _pull_remote_image(
                client, task_spec.base_remote_ref(registry), local_tag,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        work_fut = ex.submit(
            _pull_remote_image, client,
            task_spec.work_remote_ref(registry), task_spec.work_image_key,
        )
        judge_fut = ex.submit(
            _pull_remote_image, client,
            task_spec.judge_remote_ref(registry), task_spec.judge_image_key,
        )
        work_ok = work_fut.result()
        judge_ok = judge_fut.result()
    return base_ok, work_ok, judge_ok


def push_all_images(
    task_spec: TaskSpec,
    registry: str,
    client: docker.DockerClient,
) -> tuple[bool, bool, bool]:
    """Push base + work + judge images to registry. Returns (base_ok, work_ok, judge_ok).

    A per-key lock prevents redundant base image pushes when multiple
    tasks sharing the same base image are pushed in parallel.
    """
    base_key = task_spec.base_image
    with _base_push_locks_guard:
        if base_key not in _base_push_locks:
            _base_push_locks[base_key] = threading.Lock()
        lock = _base_push_locks[base_key]

    remote_ref = task_spec.base_remote_ref(registry)
    with lock:
        if remote_ref in _pushed_base_refs:
            base_ok = True
        else:
            base_ok = _push_image(client, task_spec.base_image_tag, remote_ref)
            if base_ok:
                _pushed_base_refs.add(remote_ref)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        work_fut = ex.submit(
            _push_image, client,
            task_spec.work_image_key, task_spec.work_remote_ref(registry),
        )
        judge_fut = ex.submit(
            _push_image, client,
            task_spec.judge_image_key, task_spec.judge_remote_ref(registry),
        )
        work_ok = work_fut.result()
        judge_ok = judge_fut.result()
    return base_ok, work_ok, judge_ok
