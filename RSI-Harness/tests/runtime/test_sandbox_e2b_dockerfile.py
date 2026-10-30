"""Single-stage Dockerfiles as E2B template steps (sandbox_e2b_dockerfile)."""

import io
import shlex
import tarfile
from pathlib import Path

import pytest

from rsi_harness.runtime.sandbox_contracts import SandboxError
from rsi_harness.runtime.sandbox_e2b_dockerfile import (
    extract_context,
    lines,
    parse,
    refuse_archives,
    steps,
)

ROOT = Path(__file__).parents[2]
FROM = "FROM busybox:1.37.0\n"
BASE = {"Env": ["PATH=/usr/bin:/bin"], "User": "app", "WorkingDir": "/home/app"}


def translate(dockerfile, base=BASE, build_args=None, target=None):
    plan = parse(dockerfile.encode(), build_args or {}, target)
    return steps(plan, base)


@pytest.mark.parametrize(
    "dockerfile, field, what",
    [
        (
            (
                ROOT / "sample_tasks/harbor-in-judge/tests/fix-git-build"
                "/environment/Dockerfile"
            ).read_text(),
            "dockerfile",
            "multi-stage",
        ),
        (FROM + "COPY --from=builder /a /b\n", "dockerfile", "COPY --from"),
        (FROM + "COPY --parents a /b/\n", "dockerfile", "COPY --parents"),
        (FROM + "ADD https://example.com/a.tgz /opt/\n", "dockerfile", "URL"),
        (FROM + "ADD git@github.com:a/b.git /src\n", "dockerfile", "git"),
        (FROM + "RUN --mount=type=cache,target=/c true\n", "dockerfile", "--mount"),
        (FROM + "RUN --network=none true\n", "dockerfile", "--network"),
        (FROM + "RUN <<EOF\necho hi\nEOF\n", "dockerfile", "heredoc"),
        (FROM + 'SHELL ["/bin/bash", "-c"]\n', "dockerfile", "SHELL"),
        (FROM + "ONBUILD RUN true\n", "dockerfile", "ONBUILD"),
        ("# escape=`\n" + FROM, "dockerfile", "escape"),
        ("FROM --platform=linux/arm64 busybox:1.37.0\n", "dockerfile", "--platform"),
        ("FROM scratch\n", "dockerfile", "scratch"),
        (FROM + "ENV A=${B:-b}\n", "dockerfile", "modifier"),
        (FROM + "USER app:staff\n", "dockerfile", "group"),
        (FROM + "COPY --chmod=u+x a /b\n", "dockerfile", "octal"),
        # E2B maps neither as Docker does.
        (FROM + "COPY --chown=app a /b\n", "dockerfile", "user name alone"),
        (FROM + "USER 1000\nRUN true\n", "dockerfile", "numeric USER"),
    ],
)
def test_what_e2b_steps_cannot_build_as_docker_does_is_refused(dockerfile, field, what):
    with pytest.raises(SandboxError) as refused:
        translate(dockerfile)
    assert (refused.value.code, refused.value.field) == ("unsupported", field)
    assert what in refused.value.message
    assert refused.value.message.endswith("unsupported on the e2b backend")


def test_on_modal_only_what_the_overlay_cannot_follow_is_refused():
    users = FROM + "USER 1000\nRUN id\nCOPY --chown=app a /b/\n"
    with pytest.raises(SandboxError, match="numeric USER unsupported on the e2b"):
        translate(users)
    plan = parse(users.encode(), {}, None, "modal")
    assert steps(plan, BASE, "modal")[1]["User"] == "1000"
    with pytest.raises(
        SandboxError, match="multi-stage Dockerfile unsupported on the modal"
    ):
        parse((FROM * 2).encode(), {}, None, "modal")
    with pytest.raises(
        SandboxError, match="the SHELL instruction unsupported on the modal"
    ):
        parse((FROM + 'SHELL ["bash", "-c"]\n').encode(), {}, None, "modal")


def test_modal_lines_are_the_instructions_with_their_args_resolved():
    dockerfile = (
        "ARG V=1 W\nFROM busybox:$V\nARG V\nARG W X=x Y\nARG TARGETARCH\n"
        'RUN echo "$V $W $X"\nCOPY --chown=app a* /b/\nCMD ["sh"]\n'
    )
    plan = parse(dockerfile.encode(), {"Y": "a b", "W": "w"}, None, "modal")
    assert lines(plan, steps(plan, {}, "modal")[0]) == (
        "ARG V=1",
        "ARG W=w X=x Y='a b'",
        "ARG TARGETARCH=amd64",
        'RUN echo "$V $W $X"',
        "COPY --chown=app a* /b/",
        'CMD ["sh"]',
    )


def test_modal_lines_run_a_non_root_users_run_through_setpriv():
    """Modal's builder runs RUN as root: USER is left out and a RUN under a
    non-root user (the base image's or a USER's) switches with setpriv."""
    dockerfile = FROM + "RUN a\nUSER 0\nRUN b\nUSER 65534\nRUN c $X\n"
    plan = parse(dockerfile.encode(), {}, None, "modal")
    found = steps(plan, {"User": "app", "Env": ["X=1"]}, "modal")[0]
    out = lines(plan, found)
    assert [line.split()[0] for line in out] == ["RUN", "RUN", "RUN"]
    assert out[1] == "RUN b"
    for line, user, command in (
        (out[0], "app", "export X=1 && exec /bin/sh -c a"),
        (out[2], "65534", "export X=1 && exec /bin/sh -c 'c $X'"),
    ):
        assert line.startswith(f"RUN u={user}; ")
        assert line.endswith(f" -- /bin/sh -c {shlex.quote(command)}")


def test_modal_lines_copy_what_modals_context_upload_reads(tmp_path):
    """Every COPY and ADD form becomes a COPY whose sources the pinned SDK's
    own reading picks from the context, as Docker would copy them."""
    docker_utils = pytest.importorskip("modal._utils.docker_utils")
    from modal.file_pattern_matcher import FilePatternMatcher

    dockerfile = FROM + (
        "ARG SRC=app.py\nWORKDIR /w\nCOPY ./ /all/\nADD app.py /add/\n"
        'COPY ["src", "/json/"]\nCOPY $SRC rel\n'
        "COPY --chmod=755 --chown=1:1 *.py /py/\n"
    )
    plan = parse(dockerfile.encode(), {}, None, "modal")
    copies = [
        line
        for line in lines(plan, steps(plan, {}, "modal")[0])
        if line.startswith("COPY")
    ]
    assert copies == [
        "COPY . /all/",
        "COPY app.py /add/",
        "COPY src /json/",
        "COPY app.py /w/rel",
        "COPY --chown=1:1 --chmod=755 *.py /py/",
    ]
    context = tmp_path / "context"
    (context / "src/sub").mkdir(parents=True)
    for name in ("app.py", "src/sub/a.py", ".hidden"):
        (context / name).touch()
    files = [path.relative_to(context) for path in context.rglob("*")]
    picked = []
    for line in copies:
        matcher = FilePatternMatcher(
            *docker_utils.extract_copy_command_patterns([line])
        )
        picked.append(sorted(str(path) for path in files if matcher(path)))
    assert picked == [
        [".hidden", "app.py", "src", "src/sub", "src/sub/a.py"],
        ["app.py"],
        ["src", "src/sub", "src/sub/a.py"],
        ["app.py"],
        ["app.py"],
    ]


def test_a_target_must_name_the_stage():
    assert parse(b"FROM busybox:1.37.0 AS app\n", {}, "app").base.endswith(":1.37.0")
    with pytest.raises(SandboxError) as refused:
        parse(b"FROM busybox:1.37.0 AS app\n", {}, "test")
    assert refused.value.field == "target"


def test_args_before_from_name_the_base_and_build_args_override_defaults():
    dockerfile = b"ARG TAG=1.36\nARG REPO=busybox\nFROM ${REPO}:$TAG\n"
    assert parse(dockerfile, {}, None).base == "docker.io/library/busybox:1.36"
    plan = parse(dockerfile, {"TAG": "1.37.0"}, None)
    assert plan.base == "docker.io/library/busybox:1.37.0"
    platform = b"FROM --platform=$BUILDPLATFORM busybox:1.37.0\n"
    assert parse(platform, {}, None).base == "docker.io/library/busybox:1.37.0"


def test_args_reach_run_only_and_env_reaches_later_steps_and_the_config():
    found, config = translate(
        "ARG V=0\n" + FROM + "ARG V\nARG W=2\nENV HOME_DIR=/srv/$W PATH=/opt:$PATH\n"
        "RUN echo $V $HOME_DIR\n",
        build_args={"W": "3"},
    )
    assert found[2] == ("env", {"HOME_DIR": "/srv/3", "PATH": "/opt:/usr/bin:/bin"})
    assert found[3] == (
        "run",
        "export V=0 W=3 PATH=/opt:/usr/bin:/bin HOME_DIR=/srv/3 && "
        "exec /bin/sh -c 'echo $V $HOME_DIR'",
    )
    assert config["Env"] == ["PATH=/opt:/usr/bin:/bin", "HOME_DIR=/srv/3"]


def test_relative_workdirs_and_dests_are_absolute_and_flags_become_owner_and_mode():
    found, config = translate(
        FROM + "WORKDIR app\nWORKDIR ../lib\nCOPY a b/ ./\nCOPY . sub\n"
        "ADD --chown=app:staff --chmod=0640 --link c.txt /etc/c.txt\n"
        'COPY --chown=a:b ["x", "/y/"]\n'
    )
    assert found == (
        # The base image's USER and WORKDIR open the steps, as Docker runs them.
        ("user", "app"),
        ("workdir", "/home/app"),
        ("workdir", "/home/app/app"),
        ("workdir", "/home/app/lib"),
        ("copy", "a", "/home/app/lib/", "root", None),
        ("copy", "b", "/home/app/lib/", "root", None),
        ("copy", ".", "/home/app/lib/sub", "root", None),
        ("add", "c.txt", "/etc/c.txt", "app:staff", 0o640),
        ("copy", "x", "/y/", "a:b", None),
    )
    assert config["WorkingDir"] == "/home/app/lib"


def test_exec_forms_and_an_entrypoint_resetting_the_inherited_cmd():
    base = {**BASE, "Entrypoint": ["/init"], "Cmd": ["serve"]}
    found, config = translate(
        FROM + 'RUN ["python3", "-c", "print(1)"]\nENTRYPOINT top -b\n', base
    )
    assert found[-1] == (
        "run",
        "export PATH=/usr/bin:/bin && exec python3 -c 'print(1)'",
    )
    assert config["Entrypoint"] == ["/bin/sh", "-c", "top -b"]
    assert "Cmd" not in config
    # A CMD of the Dockerfile itself stays.
    _, config = translate(FROM + 'CMD ["a"]\nENTRYPOINT ["b"]\n', base)
    assert (config["Entrypoint"], config["Cmd"]) == (["b"], ["a"])


def test_shell_forms_run_in_the_base_images_shell():
    shell = ["/bin/bash", "-o", "pipefail", "-c"]
    found, config = translate(FROM + "RUN a | b\nCMD serve\n", {**BASE, "Shell": shell})
    assert found[-1] == (
        "run",
        "export PATH=/usr/bin:/bin && exec /bin/bash -o pipefail -c 'a | b'",
    )
    assert config["Cmd"] == [*shell, "serve"]
    assert "Shell" not in config


def test_user_volume_and_ignored_instructions_set_the_config():
    found, config = translate(
        FROM + "USER 1000\nVOLUME /data\nLABEL a=b\nEXPOSE 80\nHEALTHCHECK NONE\n",
        base={},
    )
    assert found == (("user", "root"), ("workdir", "/"), ("user", "1000"))
    assert config == {"User": "1000", "Volumes": {"/data": {}}}


def test_here_strings_shifts_and_uid_0_are_not_refused():
    found, _ = translate(
        FROM + 'USER 0\nRUN cat <<<"hi"\nRUN echo $((1<<2))\n', base={}
    )
    assert [step[0] for step in found] == ["user", "workdir", "user", "run", "run"]
    assert found[2] == ("user", "root")


def build_input(tmp_path, files):
    """A build input tar: ``files`` maps a context path to (mode, data);
    data None is a directory."""
    path = tmp_path / "input.tar"
    with tarfile.open(path, "w") as archive:
        for name, (mode, data) in files.items():
            info = tarfile.TarInfo(f"d/ctx/{name}")
            info.mode = mode
            if data is None:
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return path


def test_the_extracted_context_keeps_its_modes(tmp_path):
    path = build_input(
        tmp_path,
        {"shared": (0o775, None), "shared/f": (0o775, b"x"), "s": (0o4755, b"y")},
    )
    extract_context(path, "d", tmp_path / "ctx")
    modes = {
        name: (tmp_path / "ctx" / name).stat().st_mode & 0o7777
        for name in ("shared", "shared/f", "s")
    }
    assert modes == {"shared": 0o775, "shared/f": 0o775, "s": 0o4755}


def test_an_added_archive_is_refused_only_where_the_wildcard_reaches(tmp_path):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w"):
        pass
    path = build_input(
        tmp_path, {"sub": (0o755, None), "sub/x.tar": (0o644, archive.getvalue())}
    )
    refuse_archives((("add", "*.tar", "/opt/"),), path, "d")
    with pytest.raises(SandboxError, match="local archive"):
        refuse_archives((("add", "*/*.tar", "/opt/"),), path, "d")
