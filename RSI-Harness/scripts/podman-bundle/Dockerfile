# Static Podman bundle the Harness installs for cluster image builds.
# Build with build.sh; every binary is statically linked (musl).

FROM golang:1.26-alpine3.22 AS base
RUN apk add --no-cache git make gcc g++ pkgconf musl-dev linux-headers bash \
    file autoconf automake libtool meson ninja clang llvm coreutils \
    btrfs-progs-dev libassuan-dev lvm2-dev device-mapper glib-dev glib-static \
    gpgme-dev protobuf-dev protobuf-c-dev libseccomp-dev libseccomp-static \
    libselinux-dev ostree-dev openssl go-md2man pcre2-static gettext-static \
    libcap-dev libcap-static argp-standalone json-c-dev eudev-dev
COPY fetch.sh /usr/local/bin/fetch

FROM rust:1.91-alpine3.22 AS rust
RUN apk add --no-cache git make musl-dev protoc
COPY fetch.sh /usr/local/bin/fetch
ENV RUSTFLAGS="-C link-arg=-s"

FROM base AS podman
RUN fetch podman https://github.com/containers/podman v5.8.9
WORKDIR /src/podman
ARG TAGS="seccomp selinux apparmor exclude_graphdriver_devicemapper containers_image_openpgp"
RUN set -eux; b=/out/bundle; \
    make bin/podman LDFLAGS_PODMAN="-s -w -extldflags '-static'" BUILDTAGS="$TAGS"; \
    CGO_ENABLED=0 make bin/quadlet LDFLAGS_PODMAN="-s -w -extldflags '-static'" BUILDTAGS="$TAGS"; \
    CGO_ENABLED=0 make bin/rootlessport \
      BUILDFLAGS=" -mod=vendor -ldflags=\"-s -w -extldflags '-static'\""; \
    install -D bin/podman "$b/bin/podman"; \
    install -D bin/quadlet "$b/libexec/podman/quadlet"; \
    install -D bin/rootlessport "$b/lib/podman/rootlessport"; \
    install -Dm644 completions/bash/podman "$b/share/bash-completion/completions/podman"; \
    install -Dm644 completions/zsh/_podman "$b/share/zsh/site-functions/_podman"; \
    install -Dm644 completions/fish/podman.fish "$b/share/fish/vendor_completions.d/podman.fish"; \
    install -Dm644 "$(find vendor -path '*/storage/storage.conf' | head -n1)" \
      "$b/etc/containers/storage.conf"; \
    seccomp="$(find vendor -path '*/seccomp/seccomp.json' | head -n1)"; \
    if [ -n "$seccomp" ]; then install -Dm644 "$seccomp" "$b/etc/containers/seccomp.json"; fi

FROM base AS conmon
RUN fetch conmon https://github.com/containers/conmon v2.2.1
WORKDIR /src/conmon
RUN make git-vars bin/conmon PKG_CONFIG='pkg-config --static' \
      CFLAGS='-std=c99 -Os -Wall -Wextra -Werror -static' LDFLAGS='-s -w -static' \
 && install -D bin/conmon /out/bundle/lib/podman/conmon

FROM base AS runc
RUN fetch runc https://github.com/opencontainers/runc v1.4.3
WORKDIR /src/runc
RUN make static EXTRA_LDFLAGS="-s -w" && install -D runc /out/bundle/bin/runc

FROM base AS crun
RUN fetch crun https://github.com/containers/crun 1.30.1
WORKDIR /src/crun
RUN ./autogen.sh && ./configure --disable-systemd --enable-embedded-yajl \
 && make LDFLAGS='-static-libgcc -all-static' EXTRA_LDFLAGS='-s -w' \
 && install -D crun /out/bundle/bin/crun

FROM base AS fuse-overlayfs
RUN fetch libfuse https://github.com/libfuse/libfuse fuse-3.18.3
WORKDIR /src/libfuse
RUN meson setup build --prefix /usr -D default_library=static -D examples=false \
      -D c_link_args='-lpthread -s -static' \
 && ninja -C build && touch /dev/fuse && ninja -C build install \
 && install -D /usr/bin/fusermount3 /out/bundle/bin/fusermount3
RUN fetch fuse-overlayfs https://github.com/containers/fuse-overlayfs v1.18
WORKDIR /src/fuse-overlayfs
RUN sh autogen.sh && LIBS="-ldl" LDFLAGS="-s -w -static" ./configure --prefix /usr \
 && make && install -D fuse-overlayfs /out/bundle/bin/fuse-overlayfs

FROM base AS catatonit
RUN fetch catatonit https://github.com/openSUSE/catatonit v0.2.1
WORKDIR /src/catatonit
RUN ./autogen.sh && ./configure LDFLAGS="-static" && make \
 && install -D catatonit /out/bundle/lib/podman/catatonit

FROM base AS passt
RUN fetch passt https://passt.top/passt 2026_10_02.cba3570
WORKDIR /src/passt
RUN make static && for f in passt pasta passt.avx2 pasta.avx2; do \
      if [ -f "$f" ]; then install -D "$f" "/out/bundle/bin/$f"; fi; done

FROM base AS slirp4netns
RUN fetch libslirp https://gitlab.freedesktop.org/slirp/libslirp.git v4.9.1
WORKDIR /src/libslirp
RUN meson setup build --prefix /usr -D default_library=static && ninja -C build install
RUN fetch slirp4netns https://github.com/rootless-containers/slirp4netns v1.3.6
WORKDIR /src/slirp4netns
RUN ./autogen.sh && PKG_CONFIG='pkg-config --static' LDFLAGS="-s -static" ./configure \
 && make && install -D slirp4netns /out/bundle/bin/slirp4netns

FROM rust AS netavark
RUN fetch netavark https://github.com/containers/netavark v1.17.2
WORKDIR /src/netavark
RUN cargo build --release && install -D target/release/netavark /out/bundle/lib/podman/netavark

FROM rust AS aardvark-dns
RUN fetch aardvark-dns https://github.com/containers/aardvark-dns v1.17.1
WORKDIR /src/aardvark-dns
RUN cargo build --release \
 && install -D target/release/aardvark-dns /out/bundle/lib/podman/aardvark-dns

FROM base AS assemble
COPY --from=podman /out/ /out/
COPY --from=conmon /out/ /out/
COPY --from=runc /out/ /out/
COPY --from=crun /out/ /out/
COPY --from=fuse-overlayfs /out/ /out/
COPY --from=catatonit /out/ /out/
COPY --from=passt /out/ /out/
COPY --from=slirp4netns /out/ /out/
COPY --from=netavark /out/ /out/
COPY --from=aardvark-dns /out/ /out/
COPY containers.conf policy.json registries.conf /out/bundle/etc/containers/
COPY README.bundle /out/bundle/README
WORKDIR /out/bundle
RUN set -eu; \
    for f in bin/* lib/podman/* libexec/podman/*; do \
      strip "$f"; \
      file -L "$f" | grep -Eq 'statically linked|static-pie linked' \
        || { echo "not static: $(file "$f")" >&2; exit 1; }; \
    done; \
    cat /out/versions/* | sort > VERSIONS; \
    cp -r /out/licenses licenses; \
    find . -type f | sed 's|^\./||' | sort | xargs sha256sum > /tmp/SHA256SUMS; \
    mv /tmp/SHA256SUMS SHA256SUMS; \
    bin/podman --version

FROM scratch
COPY --from=assemble /out/bundle /bundle
COPY --from=assemble /out/sources /sources
