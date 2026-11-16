#!/usr/bin/env bash
# Builds the web app into app/dspk-app-web/dist/ (served as it is: `dspk serve --web <dist>`):
#
#   index.html  the page: a loader that checks the browser (hardware drawing, WebGPU, threads)
#               before anything large is downloaded, then starts one of the builds below
#   mt/         the threaded build: drawing on web workers; a client of `dspk serve`, which
#               computes. Needs the page served cross-origin isolated (`dspk serve --web` is)
#   st/         the single-thread build: everything on the page's thread, ops on the browser's
#               WebGPU (pages not isolated: a plain file server, GitHub Pages)
#
# Each .wasm and .js is also written Brotli-compressed (.br), which `dspk serve` sends to
# browsers that accept it (the .wasm is ~25 MB, ~7 MB compressed).
#
# Needs: the pinned nightly below with rust-src and the wasm32-unknown-unknown target (gpui's web
# platform needs nightly, and threads need -Z build-std), wasm-bindgen-cli of the version in
# Cargo.lock, wasm-opt (binaryen) and brotli.
#
#   app/dspk-app-web/build.sh            both builds
#   app/dspk-app-web/build.sh mt         the threaded build only (the container's)
set -euo pipefail

# Pinned so a build can be repeated; update deliberately (and in CI with it)
TOOLCHAIN="${DSPK_WEB_TOOLCHAIN:-nightly-2026-11-14}"
WHICH="${1:-all}"

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
dist="$here/dist"
target="${CARGO_TARGET_DIR:-$root/target}"

bindgen_version=$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/version = |"/, ""); print; exit }' "$root/Cargo.lock")
bindgen=$(command -v wasm-bindgen || true)
if [ -z "$bindgen" ] || ! "$bindgen" --version | grep -q " $bindgen_version\$"; then
    bindgen=$(ls ~/.cache/.wasm-pack/wasm-bindgen-*/wasm-bindgen 2>/dev/null | while read -r b; do "$b" --version | grep -q " $bindgen_version\$" && echo "$b" && break; done || true)
fi
[ -n "$bindgen" ] || { echo "wasm-bindgen $bindgen_version not found: cargo install wasm-bindgen-cli --version $bindgen_version"; exit 1; }
wasm_opt=$(command -v wasm-opt || ls -t ~/.cache/.wasm-pack/wasm-opt-*/bin/wasm-opt 2>/dev/null | head -1 || true)
[ -n "$wasm_opt" ] || { echo "wasm-opt not found (binaryen)"; exit 1; }
command -v brotli > /dev/null || { echo "brotli not found"; exit 1; }
rustup run "$TOOLCHAIN" rustc --version > /dev/null 2>&1 || rustup toolchain install "$TOOLCHAIN" --profile minimal --component rust-src --target wasm32-unknown-unknown

# $1: mt or st; the rest: what the build needs on top of `cargo build`
build() {
    local kind=$1 dir="$dist/$1" wasm
    shift
    echo "== $kind"
    (cd "$root" && "$@")
    wasm="$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/dspk_app_web.wasm"
    rm -rf "$dir"
    "$bindgen" "$wasm" --out-dir "$dir" --target web
    "$wasm_opt" -O2 --enable-threads --enable-bulk-memory --enable-mutable-globals --enable-simd --enable-nontrapping-float-to-int --enable-sign-ext \
        "$dir/dspk_app_web_bg.wasm" -o "$dir/dspk_app_web_bg.wasm"
    find "$dir" -name '*.wasm' -o -name '*.js' | while read -r f; do brotli -f -q 11 "$f" -o "$f.br"; done
    du -h "$dir/dspk_app_web_bg.wasm" "$dir/dspk_app_web_bg.wasm.br" | sed 's/^/   /'
}

mkdir -p "$dist"
cp "$here/index.html" "$dist/index.html"

if [ "$WHICH" = all ] || [ "$WHICH" = mt ]; then
    # Shared memory and atomics, and a standard library built with them (-Z build-std); the
    # exports wasm-bindgen's thread support needs. No compute feature: wgpu cannot cross threads
    export CARGO_TARGET_DIR="$target/web-mt"
    RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128' \
    build mt rustup run "$TOOLCHAIN" cargo rustc -p dspk-app-web --lib --no-default-features --target wasm32-unknown-unknown --release \
        -Z build-std=std,panic_abort -- \
        -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=4294967296 \
        -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base
fi
if [ "$WHICH" = all ] || [ "$WHICH" = st ]; then
    export CARGO_TARGET_DIR="$target/web-st"
    RUSTFLAGS='-C target-feature=+simd128' \
    build st rustup run "$TOOLCHAIN" cargo build -p dspk-app-web --lib --target wasm32-unknown-unknown --release
fi
echo "Built into $dist (serve it: dspk serve --web $dist)"
