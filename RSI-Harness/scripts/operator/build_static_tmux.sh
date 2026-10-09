#!/usr/bin/env bash
# Build a statically linked tmux for env services that lack one
# (docs/sandbox-operator-guide.md, "tmux for offline envs").
#
#   scripts/operator/build_static_tmux.sh OUTPUT
#
# Builds the pinned tmux release against musl, libevent and an ncurses
# with compiled-in terminfo fallbacks, in a throwaway container of the
# pinned Alpine image (pulled if absent, then kept); its stdout carries the
# stripped binary, which lands at OUTPUT (mode 0755). Prints the
# `sha256  OUTPUT` line to put in [environments.host.tmux]. The build
# needs network for the Alpine packages and the source tarballs, whose
# SHA-256 is checked; RSI_TMUX_BUILD_IMAGE overrides the image (pin it by
# digest). Nothing else is left behind: no image, no build cache. Run it as
# a user that may use Docker; the result runs on glibc and musl images.
set -euo pipefail

case "${1:-}" in
  -h | --help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "" | -*) echo "usage: $0 OUTPUT" >&2; exit 2 ;;
esac
output=$1
image=${RSI_TMUX_BUILD_IMAGE:-alpine:3.21@sha256:ce64758a109eb420d874a118f87920e625e12d3634e03b4a5573fd9f6e5d3507}
tmux_version=3.5a
tmux_sha256=16216bd0877170dfcc64157085ba9013610b12b082548c7c9542cc0103198951
ncurses_version=6.5
ncurses_sha256=136d91bc269a9a5785e5f9e980bc76ab57428f604ce3e5a5a90cebc767971cc6
# Compiled into the binary: tmux finds these TERMs without any terminfo
# database in the image (an image's own database is still read first).
fallbacks=xterm-256color,xterm,tmux-256color,tmux,screen-256color,screen,vt100,linux,dumb

# Runs in the container as root; only the binary goes to stdout.
build=$(
  cat <<'EOF'
set -eu
exec 3>&1 1>&2
apk add --no-cache build-base bison pkgconf libevent-dev libevent-static \
  ncurses ncurses-terminfo
cd /tmp
wget -q -O tmux.tar.gz \
  "https://github.com/tmux/tmux/releases/download/$TMUX_VERSION/tmux-$TMUX_VERSION.tar.gz"
wget -q -O ncurses.tar.gz \
  "https://invisible-mirror.net/archives/ncurses/ncurses-$NCURSES_VERSION.tar.gz"
printf '%s  tmux.tar.gz\n%s  ncurses.tar.gz\n' "$TMUX_SHA256" "$NCURSES_SHA256" |
  sha256sum -c -
tar xzf ncurses.tar.gz
tar xzf tmux.tar.gz
cd "/tmp/ncurses-$NCURSES_VERSION"
./configure --prefix=/opt/ncurses --without-shared --with-normal \
  --without-debug --without-ada --without-cxx --without-cxx-binding \
  --without-manpages --without-progs --without-tests --disable-widec \
  --enable-overwrite --disable-db-install --with-fallbacks="$FALLBACKS" \
  --with-terminfo-dirs=/etc/terminfo:/lib/terminfo:/usr/share/terminfo \
  --with-default-terminfo-dir=/usr/share/terminfo \
  --with-tic-path=/usr/bin/tic --with-infocmp-path=/usr/bin/infocmp
make -j"$(nproc)"
make install
cd "/tmp/tmux-$TMUX_VERSION"
./configure --enable-static --prefix=/usr/local \
  CPPFLAGS=-I/opt/ncurses/include LDFLAGS=-L/opt/ncurses/lib \
  LIBNCURSES_CFLAGS=-I/opt/ncurses/include LIBNCURSES_LIBS=-lncurses
make -j"$(nproc)"
strip tmux
./tmux -V
cat tmux >&3
EOF
)

partial="$output.partial.$$"
trap 'rm -f "$partial"' EXIT
docker run --rm --network bridge --label rsi.operator=build-static-tmux \
  -e TMUX_VERSION="$tmux_version" -e TMUX_SHA256="$tmux_sha256" \
  -e NCURSES_VERSION="$ncurses_version" -e NCURSES_SHA256="$ncurses_sha256" \
  -e FALLBACKS="$fallbacks" \
  "$image" sh -c "$build" >"$partial"
chmod 0755 "$partial"
"$partial" -V >&2
mv -f "$partial" "$output"
sha256sum "$output"
