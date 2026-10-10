# A6: a build RUN step sees no Docker, containerd or BuildKit socket at all
# (/run/buildkit does not exist; the broker unlinks buildkitd's trace
# collector socket before any build, so /dev/otel-grpc.sock is absent too).
FROM public.ecr.aws/docker/library/alpine:3.21
RUN test ! -e /run/buildkit \
    && sockets=$(find / \( -path /proc -o -path /sys \) -prune -o -type s -print) \
    && echo "RUN step sockets: ${sockets:-none}" \
    && test -z "$sockets"
