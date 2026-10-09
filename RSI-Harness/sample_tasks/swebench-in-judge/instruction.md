This is a sample of SWE-bench Verified judged inside RSI Harness, fully
offline: submit with `rsi-submit`. The Judge's fixed procedure runs
Harbor's `oracle` and `nop` agents over three SWE-bench Verified tasks
shipped in `/tests/swebench-verified`, each in an environment without
network that the host broker creates from the task's pre-pulled image, and
scores 1 only if every oracle trial scores 1 and every nop trial scores 0.

The tasks are Harbor's own (registry dataset `swebench-verified@1.0`),
vendored with three changes so that no step needs the network: each names
its image by digest (`docker_image`, `network_mode = "no-network"`), its
`tests/test.sh` does not reinstall the repository with pip (the image holds
the editable install), and it grades with a standard-library copy of
SWE-bench's pytest log parser instead of fetching `swebench` from PyPI.

Harbor is installed here too. To run the tasks yourself before submitting:

    PYTHONPATH=$RSI_SANDBOX_PYTHONPATH harbor run \
      --env rsi_sandbox_harbor:ManagedSandboxEnvironment \
      -a oracle -p /path/to/swebench-verified
