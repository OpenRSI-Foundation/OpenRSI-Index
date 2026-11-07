"""Install public phase clients into the task image at build time."""

from pathlib import Path


def install():
    from rsi_harness import judge_agent
    from rsi_harness.cluster.bluevela import netns_relay
    from rsi_harness.integrations.submit_client import generate_submit_client
    from rsi_loop.harness.agent import oracle

    for name, content in (
        ("rsi-submit", generate_submit_client()),
        (
            "rsi-isolate",
            (Path(judge_agent.__file__).parent / "rsi-isolate").read_text(),
        ),
        ("rsi-netns-relay", Path(netns_relay.__file__).read_text()),
        (
            "rsi-oracle",
            (Path(oracle.__file__).parent / "oracle" / "rsi-oracle").read_text(),
        ),
    ):
        path = Path("/usr/local/bin") / name
        path.write_text(content)
        path.chmod(0o755)
    Path("/etc/codex").mkdir(exist_ok=True)
    # No setuid executables are needed by Work after namespace setup.


if __name__ == "__main__":
    install()
