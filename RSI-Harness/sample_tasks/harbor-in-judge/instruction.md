This is a sample of Harbor running inside RSI Harness: submit with
`rsi-submit`. The Judge's fixed procedure runs Harbor's `oracle` and `nop`
agents over a Terminal-Bench 2 subset shipped in `/tests`, each task in an
environment the host broker creates, and scores 1 only if every oracle
trial scores 1 and every nop trial scores 0.

Harbor is installed here too. To run a suite yourself before submitting:

    PYTHONPATH=$RSI_SANDBOX_PYTHONPATH harbor run \
      --env rsi_sandbox_harbor:ManagedSandboxEnvironment \
      -a oracle -p <task or dataset directory>
