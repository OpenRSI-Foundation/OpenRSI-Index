#!/bin/sh
# fetch NAME URL REF: check REF out at /src/NAME and keep its source,
# licenses and commit under /out.
set -eu
name=$1 url=$2 ref=$3
git clone -q -c advice.detachedHead=false --depth=1 --branch "$ref" \
  --recurse-submodules --shallow-submodules "$url" "/src/$name"
mkdir -p /out/sources "/out/licenses/$name" /out/versions
tar -C /src --exclude=.git -czf "/out/sources/$name-$ref.tar.gz" "$name"
find "/src/$name" -maxdepth 1 \( -iname 'LICENSE*' -o -iname 'COPYING*' \) \
  -exec cp {} "/out/licenses/$name/" \;
echo "$name $ref $url $(git -C "/src/$name" rev-parse HEAD)" > "/out/versions/$name"
