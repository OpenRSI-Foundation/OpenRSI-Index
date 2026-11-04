"""Real Docker output-failure containment, including Docker Desktop controllers.

RSI_ACCEPTANCE=1 and RSI_DOCKER_TEST_IMAGE opt in. The existing Linux image
must contain /bin/sh and sleep. Only containers created by this test are removed.
"""

import errno
import os
import time
import uuid

import docker
import pytest

from rsi_harness.errors import InfrastructureError
from rsi_harness.models import ContainerRef
from rsi_harness.runtime import docker as module
from tests.runtime.test_docker import make_runtime

pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize("redact", [False, True])
@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EDQUOT, errno.EIO])
def test_real_docker_stops_only_the_container_with_failed_output(
    tmp_path, monkeypatch, redact, error_number
):
    image = os.environ.get("RSI_DOCKER_TEST_IMAGE")
    if not image:
        pytest.skip("set RSI_DOCKER_TEST_IMAGE to an existing Linux image")
    client = docker.from_env()
    owned = []
    try:
        for _ in range(2):
            owned.append(
                client.containers.run(
                    image,
                    ["/bin/sh", "-c", "sleep 300"],
                    detach=True,
                    network_disabled=True,
                    read_only=True,
                    labels={"rsi-output-acceptance": uuid.uuid4().hex},
                )
            )
        candidate, other = owned
        runtime = make_runtime(client, tmp_path, role="judge")
        output = tmp_path / "engine-root/output.log"
        ref = ContainerRef(container_id=candidate.id, role="judge")
        good = runtime.exec(
            ref,
            ("/bin/sh", "-c", "printf 'original\\n'"),
            output_path=output,
            redact_output=redact,
            timeout_seconds=20,
        )
        assert good.full_output_captured and output.read_text() == "original\n"

        def fail(*_args):
            raise OSError(error_number, "injected output storage failure")

        writer = module._RedactedOutputWriter if redact else module._RawOutputWriter
        monkeypatch.setattr(writer, "append", fail)
        started = time.monotonic()
        with pytest.raises(InfrastructureError, match="output capture failed"):
            runtime.exec(
                ref,
                ("/bin/sh", "-c", "printf 'new\\n'; sleep 120"),
                output_path=output,
                redact_output=redact,
                timeout_seconds=120,
            )
        assert time.monotonic() - started < 20
        candidate.reload()
        other.reload()
        assert not candidate.attrs["State"]["Running"]
        assert other.attrs["State"]["Running"]
        assert output.read_text() == "original\n"
    finally:
        for container in owned:
            container.remove(force=True)
        client.close()
