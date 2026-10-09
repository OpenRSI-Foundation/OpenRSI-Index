# Shared by the scripted Work agents of the operator acceptance; sourced
# after unpacking $RSI_ACCEPTANCE_FILES into $root. Results are lines
# "RSI-ACCEPTANCE <key> <value>" in the Agent output, which
# scripts/operator/sandbox_acceptance.sh reads.

note() {
    printf 'RSI-ACCEPTANCE %s\n' "$*"
}

submit() {
    rsi-submit
    note "submit-exit $?"
}

# The Work half of A6: the only socket is the endpoint's and it is no
# Engine API; a build RUN step and a child see no socket at all.
probe_work() {
    python3 "$root/work/probe_endpoint.py"
    if rsi-sandbox build "$root/work/run-probe" --no-cache --timeout 900 \
        > "$root/run-probe.json"
    then
        note "a6-run-step pass"
        rsi-sandbox image-rm "$(python3 -c 'import json, sys
print(json.load(open(sys.argv[1]))["result"]["image"]["handle"])' "$root/run-probe.json")" > /dev/null
    else
        note "a6-run-step fail"
    fi
    local compose=(rsi-sandbox compose -f "$root/work/probe-compose.yaml" --network none)
    if "${compose[@]}" up > /dev/null; then
        local found
        found=$("${compose[@]}" exec main -- \
            find / '(' -path /proc -o -path /sys ')' -prune -o -type s -print)
        note "a6-child-sockets ${found:-none}"
        "${compose[@]}" down > /dev/null
    else
        note "a6-child-sockets error"
    fi
}
