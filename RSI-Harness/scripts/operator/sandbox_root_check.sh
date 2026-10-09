#!/usr/bin/env bash
# Root-only checks of the managed Docker capability (spec 8 step 5, all
# eight items; the M2-M8 items of sandbox-root-checks.md; R11 the allowlist
# network mode). Each check
# announces itself, runs as root against its own run ids and prints
# PASS/FAIL; a table follows.
#
#   sudo scripts/operator/sandbox_root_check.sh [--dry-run] [--only R1,R4,...]
#       [--skip-acceptance] [--scratch DIR]
#
# R10 is the acceptance A1-A8 (sandbox_acceptance.sh, hours); spec step 5
# includes it, --skip-acceptance leaves it out (then R10 is SKIPPED, and
# the table says the root check is incomplete).
# A check only creates, and removes by exact label or derived name, objects
# of the runs it starts: rsi-* containers, volumes, bridges, rules
# (rsi-<run>-... jumps, RSI_F_/RSI_I_/RSI_A_ chains of those rules), loop files
# under its own data root, built rsi-sbx-img tags. Nothing of another run
# or user is touched; R9 compares a before/after snapshot of rsi objects,
# read only, and fails only on the checks' own runs (test run ids m<N>-...,
# and the acceptance runs under the scratch directory); objects of other
# runs that started meanwhile are listed apart. The D tests re-run here
# switch to the real firewall and a loop-ext4 builder in root mode
# (RSI_SANDBOX_ROOT_MODE=1).
# The scratch directory is new and root's: by default `mktemp -d`, and a
# --scratch DIR must not exist yet (its parent must).
# --dry-run (any user) prints every check and command without running one.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${RSI_PYTHON:-$REPO/.venv/bin/python}
SCRATCH=
DRY_RUN=0
ONLY=
ACCEPTANCE=1
IDS=(R1 R2 R3 R4 R5 R6 R7 R8 R11 R10 R9)

usage() {
    sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while (($#)); do
    case $1 in
        --dry-run) DRY_RUN=1 ;;
        --only) ONLY=${2:?}; shift ;;
        --skip-acceptance) ACCEPTANCE=0 ;;
        --scratch) SCRATCH=${2:?}; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
    shift
done
for id in ${ONLY//,/ }; do
    [[ " ${IDS[*]} " == *" $id "* ]] || {
        echo "unknown check in --only: $id (one of: ${IDS[*]})" >&2
        exit 2
    }
done
export PYTHONDONTWRITEBYTECODE=1 RSI_ACCEPTANCE=1 RSI_SANDBOX_ROOT_MODE=1
export RSI_REQUIRE_SANDBOX_INTEGRATION=1
PYTEST=("$PY" -m pytest -q -rA -p no:cacheprovider --basetemp)
RESULTS=()

selected() {
    [[ -z $ONLY ]] || [[ ,$ONLY, == *,$1,* ]]
}

# check ID TITLE COMMAND...: announce, run (not with --dry-run), record.
check() {
    local id=$1 title=$2
    shift 2
    selected "$id" || return 0
    printf '\n== %s %s\n   $' "$id" "$title"
    printf ' %q' "$@"
    printf '\n'
    if ((DRY_RUN)); then
        RESULTS+=("$id|DRY-RUN|$title")
        return 0
    fi
    local log=$SCRATCH/$id.log
    if (cd "$REPO" && "$@") > "$log" 2>&1; then
        RESULTS+=("$id|PASS|$title")
        tail -n 3 "$log" | sed 's/^/   /'
    else
        RESULTS+=("$id|FAIL|$title (log: $log)")
        tail -n 25 "$log" | sed 's/^/   /'
    fi
}

# pytest NODE...: the D/acceptance tests as root, each run with its own
# pytest base temp under the scratch directory.
pytest_check() {
    local id=$1 title=$2
    shift 2
    check "$id" "$title" "${PYTEST[@]}" "$SCRATCH/tmp-$id" "$@"
}

# The scratch directory: created here, by root, and never one that already
# existed (another user could have planted links in it).
make_scratch() {
    if [[ -z $SCRATCH ]]; then
        SCRATCH=$(mktemp -d /var/tmp/rsi-rc-XXXXXX) || exit 2
    elif ! mkdir -m 0700 -- "$SCRATCH"; then
        echo "--scratch $SCRATCH must not exist yet (its parent must)" >&2
        exit 2
    fi
    if [[ -L $SCRATCH || $(stat -c %u -- "$SCRATCH") != 0 ]]; then
        echo "scratch $SCRATCH is not root's own directory" >&2
        exit 2
    fi
    chmod 0700 -- "$SCRATCH"
}

if ((!DRY_RUN)) && [[ $(id -u) != 0 ]]; then
    echo "run as root (sudo $0); --dry-run works as any user" >&2
    exit 2
fi
[[ -x $PY ]] || { echo "missing $PY" >&2; exit 2; }
if ((DRY_RUN)); then
    SCRATCH=${SCRATCH:-/var/tmp/rsi-rc-XXXXXX}
else
    make_scratch
fi
printf '== scratch %s; root mode: real iptables firewall, loop-ext4 builders\n' \
    "$SCRATCH"
# R5 kills through pidfd with this interpreter: production's Anaconda 3.13
# lacks os.pidfd_open, so its libc/raw-syscall fallback is what runs.
PIDFD=$("$PY" -c 'import os, signal, sys
native = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
print(sys.executable, sys.version.split()[0],
      "os.pidfd_open" if native else "libc/syscall pidfd fallback")')
printf '== interpreter %s\n' "$PIDFD"
if ((!DRY_RUN)); then
    (cd "$REPO" && "$PY" -m tests.acceptance.audit host-snapshot) \
        > "$SCRATCH/before.json"
fi

ROOT=tests/acceptance/test_root_checks.py
pytest_check R1 "public env: internet and DNS reachable; metadata, RFC1918, LAN, \
gateway (INPUT), sibling env and Work-style bridge blocked (spec 8.1, M2 R1)" \
    "$ROOT::test_r1_a_public_env_reaches_the_internet_and_nothing_private"
pytest_check R2 "services reach each other by alias (public and none), another \
env never; without the intra-bridge ACCEPT peers fail (spec 8.2, M2 R2, M3)" \
    "$ROOT::test_r2_services_reach_each_other_by_alias_and_never_another_env" \
    "$ROOT::test_r2_without_the_intra_bridge_accept_peers_are_rejected"
pytest_check R3 "builder RUN step: metadata, RFC1918, LAN, gateway and a child \
blocked; https://pypi.org reachable; BuildKit's own fetches (ADD <url>, git \
source, FROM) reach public egress only (spec 8.3, M8)" \
    "$ROOT::test_r3_builder_run_steps_reach_public_egress_only"
pytest_check R4 "loop-ext4 builder state: ENOSPC is 'disk' and the builder \
lives; every build path with the real builder firewall; kill -9 at \
builder/build/load/loaded then recover: losetup -d, file, bridge, rule gone, \
tags and dangling images swept (spec 8.4, M8)" \
    tests/integration/test_sandbox_build_docker.py
pytest_check R5 "exec group kill through pidfd (interrupt, timeout TERM->KILL, \
setsid survives, OOM 137) and cgroup.kill of a paused env with a sentinel \
(spec 8.5, M3, M4; $PIDFD)" \
    tests/integration/test_sandbox_exec_docker.py \
    "$ROOT::test_r5_a_group_kill_spares_a_setsid_process_and_other_execs" \
    "$ROOT::test_r5_a_paused_env_is_killed_through_cgroup_kill"
pytest_check R6 "kill -9 mid env_create/pull and with the Work group paused, \
then recover with the real firewall (a paused env through cgroup.kill, no \
write after the freeze); recovery sees the whole /proc (spec 8.6, M6)" \
    tests/integration/test_sandbox_env_recovery.py \
    "$ROOT::test_r6_recovery_sees_every_process_of_the_host"
pytest_check R7 "env backend and brokered envs as root: real firewall on env \
bridges (every rule's jumps and chains gone), disk measurements, paused \
services killed by cgroup.kill (M3, M5)" \
    tests/integration/test_sandbox_env_docker.py \
    tests/integration/test_sandbox_envs_docker.py
pytest_check R8 "Harbor plugin with the real firewall: prebuilt, Compose \
sidecar, in a container with only the endpoint, the sample's fixed procedure, \
a 1 GiB directory round trip (M7)" \
    tests/integration/test_harbor_env_trial.py \
    tests/integration/test_harbor_in_container.py \
    tests/integration/test_harbor_in_judge_sample.py \
    "$ROOT::test_m7_a_gib_directory_round_trips_through_the_plugin"
pytest_check R11 "allowlist env: a listed hostname (port-limited) and IP \
reached, their names resolved by the embedded DNS; unlisted hosts, a direct \
resolver, listed metadata/RFC1918/LAN and the gateway (INPUT) blocked; an \
operator-approved private CIDR reached; a refresh replaces the allow chain \
in place and attests exactly" \
    "$ROOT::test_r11_an_allowlist_env_reaches_its_entries_and_nothing_else"
if ((ACCEPTANCE)); then
    check R10 "A1-A8 through rsi-harness run/recover (spec 8.7)" \
        scripts/operator/sandbox_acceptance.sh --scratch "$SCRATCH/acc"
elif selected R10; then
    printf '\n== R10 A1-A8 skipped (--skip-acceptance)\n'
    RESULTS+=("R10|SKIPPED|A1-A8 through rsi-harness run/recover (spec 8.7): \
--skip-acceptance, the root check is incomplete")
fi
# Last: nothing any check made is left (spec 8.8).
check R9 "audit: no new rsi containers, volumes, networks, rsi-sbx-img tags, \
RSI_ chains or rsi- jumps, rsi bridges or sb/build loop files of the checks' \
runs since the start (other runs' listed apart)" \
    "$PY" -m tests.acceptance.audit host-diff "$SCRATCH/before.json" \
    --runs-under "$SCRATCH"

printf '\n%-5s %-8s %s\n' CHECK RESULT DESCRIPTION
failed=0
for row in "${RESULTS[@]}"; do
    IFS='|' read -r id result title <<< "$row"
    printf '%-5s %-8s %s\n' "$id" "$result" "$title"
    [[ $result == FAIL ]] && failed=1
done
if ((${#RESULTS[@]} == 0)); then
    echo "no check ran" >&2
    failed=1
fi
((DRY_RUN)) && echo "dry run: nothing was run"
exit "$failed"
