#!/usr/bin/env bash
# Smoke test of the dspk server image, end to end, with nothing on the host but Podman: the image
# is the server and the client.
#
#   build → doctor → data → serve → enroll → receive (and refusals) → sort → fetch → stop
#
# Usage (from the repository root):
#   deploy/smoke-test.sh                 # target picked from the host: nvidia (CDI), mesa (/dev/dri), cpu
#   deploy/smoke-test.sh --target cpu    # or nvidia, mesa
#   deploy/smoke-test.sh --no-build      # test the image already built
#
# Settings (environment, with their defaults):
#   PORT=50079                   UDP port on the host
#   SORTER=mountainsort5 PROBE=tetrode CHANNELS=4 DURATION=20  the synthetic recording (s) and its sort
#   WORK=$XDG_CACHE_HOME/dspk-smoke-test   working folder (removed on success; kept on failure)
# Exits non-zero at the first failure; its containers are always removed.
set -euo pipefail

TARGET=""
BUILD=1
while [ $# -gt 0 ]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --no-build) BUILD=0; shift ;;
        -h | --help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
PORT="${PORT:-50079}"
SORTER="${SORTER:-mountainsort5}"
PROBE="${PROBE:-tetrode}"
CHANNELS="${CHANNELS:-4}"
DURATION="${DURATION:-20}"
WORK="${WORK:-${XDG_CACHE_HOME:-$HOME/.cache}/dspk-smoke-test}"
NAME="dspk-smoke-$$"
# The working folder is removed and recreated: refuse anything not named for this test
case "$(basename "$WORK")" in *dspk-smoke*) ;; *) echo "WORK must be a folder named *dspk-smoke* (it is deleted): $WORK" >&2; exit 2 ;; esac

if [ -z "$TARGET" ]; then
    if [ -e /etc/cdi/nvidia.yaml ] || [ -e /var/run/cdi/nvidia.yaml ]; then TARGET=nvidia
    elif [ -e /dev/dri ]; then TARGET=mesa
    else TARGET=cpu; fi
fi
IMAGE="localhost/dspk-serve:$TARGET"
case "$TARGET" in
    # GPU containers: SELinux labelling off for this container only (deploy/README.md, GPU)
    nvidia) GPU=(--device nvidia.com/gpu=all --security-opt=label=disable); RUNTIME=cuda ;;
    mesa) GPU=(--device /dev/dri --group-add keep-groups --security-opt=label=disable); RUNTIME=wgpu ;;
    cpu) GPU=(); RUNTIME=cpu ;;
    *) echo "unknown target $TARGET (nvidia, mesa, cpu)" >&2; exit 2 ;;
esac
# The hardening of deploy/README.md: you on the host = the container's dspk user
HARDENED=(--read-only --cap-drop=all --security-opt=no-new-privileges --pids-limit=1024 --userns=keep-id:uid=10001,gid=10001)

step=0
pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1" >&2; [ $# -gt 1 ] && printf '      %s\n' "$2" >&2; echo "Working folder kept: $WORK" >&2; exit 1; }
next() { step=$((step + 1)); printf '\n[%d] %s\n' "$step" "$1"; }
cleanup() { podman rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# The image as a client: host network (reaches the published port), identities read-only
client() {
    podman run --rm --network host "${HARDENED[@]}" -v "$WORK/ids:/ids:ro,Z" -v "$WORK/fetched:/fetched:Z" "$IMAGE" "$@"
}

echo "dspk smoke test: target $TARGET (runtime $RUNTIME), image $IMAGE, port $PORT"
rm -rf "$WORK" && mkdir -p "$WORK/data/smoke" "$WORK/ids" "$WORK/fetched"
chmod 700 "$WORK/ids"

next "build"
if [ "$BUILD" = 1 ]; then
    [ -f deploy/Containerfile ] || fail "build" "run from the repository root"
    podman build -f deploy/Containerfile --ignorefile deploy/.containerignore --target "$TARGET" -t "$IMAGE" . > "$WORK/build.log" 2>&1 \
        || fail "build" "see $WORK/build.log"
    pass "built $IMAGE"
else
    podman image exists "$IMAGE" || fail "build" "$IMAGE does not exist (drop --no-build)"
    pass "using $IMAGE as built"
fi

next "doctor"
podman run --rm "${HARDENED[@]}" "${GPU[@]}" "$IMAGE" doctor > "$WORK/doctor.txt" 2>&1 || fail "doctor" "$(tail -3 "$WORK/doctor.txt")"
grep -E "^\s+$RUNTIME\s+usable" "$WORK/doctor.txt" > /dev/null || fail "doctor: $RUNTIME is not usable" "$(grep -E '^\s+(wgpu|cuda|cpu)' "$WORK/doctor.txt" | tr -s ' ')"
pass "$RUNTIME usable ($(grep -cE '^\s+\w+\s+usable' "$WORK/doctor.txt") runtime(s) usable)"

next "data"
podman run --rm "${HARDENED[@]}" -v "$WORK/data:/data:Z" "$IMAGE" generate /data/smoke/rec.bin --channels "$CHANNELS" --duration "$DURATION" > "$WORK/generate.log" 2>&1 \
    || fail "data" "$(tail -3 "$WORK/generate.log")"
[ -s "$WORK/data/smoke/rec.bin" ] || fail "data" "no recording written"
pass "synthetic recording: $CHANNELS channels × $DURATION s"

next "serve"
podman run -d --name "$NAME" "${HARDENED[@]}" "${GPU[@]}" -e DSPK_RUNTIME="$RUNTIME" -p "$PORT:50051/udp" \
    -v "$WORK/data:/data:Z" "$IMAGE" serve --bind 0.0.0.0:50051 --max-clients 4 > /dev/null || fail "serve" "the container did not start"
for _ in $(seq 1 30); do
    podman logs "$NAME" 2>&1 | grep -q "Listening on" && break
    sleep 1
done
podman logs "$NAME" 2>&1 | grep -q "Listening on" || fail "serve" "$(podman logs "$NAME" 2>&1 | tail -3)"
pass "listening on UDP $PORT, data root /data"

next "enroll"
(umask 077; podman exec "$NAME" dspk clients enroll viewer --out - > "$WORK/ids/viewer.pem" 2> /dev/null) || fail "enroll viewer"
(umask 077; podman exec "$NAME" dspk clients enroll lab --control --out - > "$WORK/ids/lab.pem" 2> /dev/null) || fail "enroll lab"
grep -q "PRIVATE KEY" "$WORK/ids/lab.pem" || fail "enroll" "no key in the identity file"
[ "$(stat -c %a "$WORK/ids/lab.pem")" = 600 ] || fail "enroll" "identity file is not mode 600"
pass "viewer (read) and lab (control) enrolled, identities 0600, none left on the server"
if podman exec "$NAME" sh -c 'ls /data/*.pem /data/**/*.pem' > /dev/null 2>&1; then fail "enroll" "an identity file is under /data"; fi

next "receive"
client receive "127.0.0.1:$PORT" --identity /ids/viewer.pem --open smoke/rec.bin --max-frames 3 > "$WORK/receive.txt" 2>&1 \
    || fail "receive" "$(tail -3 "$WORK/receive.txt")"
grep -q "Received 3 frames" "$WORK/receive.txt" || fail "receive" "$(tail -3 "$WORK/receive.txt")"
pass "viewer received 3 frames of smoke/rec.bin"
if client receive "127.0.0.1:$PORT" --insecure --max-frames 1 > /dev/null 2>&1; then fail "refusal" "a client without an identity got in"; fi
pass "refused: a client without an identity"
if client remote "127.0.0.1:$PORT" --identity /ids/viewer.pem sort smoke/rec.bin --sorter "$SORTER" --probe "$PROBE" > /dev/null 2>&1; then
    fail "refusal" "a read-only client started a sort"
fi
pass "refused: a sort from the read-only client"
if client remote "127.0.0.1:$PORT" --identity /ids/viewer.pem fetch .dspk-auth/ca.key.pem /fetched/key > /dev/null 2>&1; then
    fail "refusal" "the server's CA key was fetched"
fi
pass "refused: fetching the server's key"

next "sort"
client remote "127.0.0.1:$PORT" --identity /ids/lab.pem sort smoke/rec.bin --sorter "$SORTER" --probe "$PROBE" --wait > "$WORK/sort.txt" 2>&1 \
    || fail "sort" "$(tail -4 "$WORK/sort.txt")"
grep -qE "[0-9]+ units, [0-9]+ spikes" "$WORK/sort.txt" || fail "sort" "$(tail -4 "$WORK/sort.txt")"
pass "$SORTER on the server: $(grep -oE '[0-9]+ units, [0-9]+ spikes in [0-9.]+ s' "$WORK/sort.txt" | tail -1)"
client remote "127.0.0.1:$PORT" --identity /ids/viewer.pem fetch "smoke/rec.$SORTER.sorting.zarr" /fetched/result.sorting.zarr > "$WORK/fetch.txt" 2>&1 \
    || fail "fetch" "$(tail -3 "$WORK/fetch.txt")"
[ -f "$WORK/fetched/result.sorting.zarr/zarr.json" ] || fail "fetch" "no zarr.json in the fetched sorting"
diff -r "$WORK/data/smoke/rec.$SORTER.sorting.zarr" "$WORK/fetched/result.sorting.zarr" > /dev/null || fail "fetch" "the fetched sorting differs from the server's"
pass "fetched the sorting: identical ($(find "$WORK/fetched/result.sorting.zarr" -type f | wc -l) files)"
grep -q '"action":"start"' "$WORK/data/.dspk-auth/audit.jsonl" || fail "audit" "the sort is not in the audit log"
pass "audit log: $(wc -l < "$WORK/data/.dspk-auth/audit.jsonl") control requests recorded"

next "stop"
start=$(date +%s)
podman stop -t 10 "$NAME" > /dev/null || fail "stop"
took=$(($(date +%s) - start))
podman logs "$NAME" 2>&1 | grep -q "^Stopping" || fail "stop" "the server did not stop cleanly (no 'Stopping')"
[ "$took" -lt 10 ] || fail "stop" "took $took s (killed at the timeout?)"
pass "stopped cleanly in ${took} s"

rm -rf "$WORK"
printf '\nAll passed (target %s).\n' "$TARGET"
