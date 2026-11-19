#!/usr/bin/env bash
# Builds the web app into app/dspk-app-web/dist/ (served as it is: `dspk serve --web <dist>`):
#
#   index.html  the page: a loader that checks the browser (hardware drawing, WebGPU, threads)
#               before anything large is downloaded, then starts the app
#   mt/         the app, threaded: drawing on web workers. Needs the page cross-origin isolated
#               (`dspk serve --web` sends the headers; elsewhere isolate-sw.js adds them)
#   engine/     the engine (dspk-engine-web) for a page that runs alone (GitHub Pages): a
#               single-thread module in a worker of its own, which owns WebGPU, reads the files
#               the user picked and runs a view's ops; a page `dspk serve` served does not use it
#   isolate-sw.js  the service worker that isolates the page where the host cannot
#
# Each .wasm and .js is also written Brotli-compressed (.br), which `dspk serve` sends to
# browsers that accept it (the .wasm is ~25 MB, ~7 MB compressed).
#
# Needs: the pinned nightly below with rust-src and the wasm32-unknown-unknown target (gpui's web
# platform needs nightly, and threads need -Z build-std), wasm-bindgen-cli of the version in
# Cargo.lock, wasm-opt (binaryen) and brotli.
#
#   app/dspk-app-web/build.sh            the app and the engine (GitHub Pages)
#   app/dspk-app-web/build.sh mt         the app only (the container's: its server computes)
set -euo pipefail

# zarrs' zstd and Blosc codecs are C, compiled for wasm32 by clang (no C library is needed at run
# time: the module imports nothing but wasm-bindgen's glue)
export CC_wasm32_unknown_unknown="${CC_wasm32_unknown_unknown:-clang}"
export AR_wasm32_unknown_unknown="${AR_wasm32_unknown_unknown:-llvm-ar}"
command -v "$CC_wasm32_unknown_unknown" > /dev/null || { echo "clang not found (zarrs' C codecs for wasm32)"; exit 1; }

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

# $1: the folder under dist/; $2: the crate's library name; the rest: the cargo command
build() {
    local kind=$1 lib=$2 dir="$dist/$1" wasm
    shift 2
    echo "== $kind"
    (cd "$root" && "$@")
    wasm="$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/$lib.wasm"
    rm -rf "$dir"
    "$bindgen" "$wasm" --out-dir "$dir" --target web
    "$wasm_opt" -O2 --enable-threads --enable-bulk-memory --enable-mutable-globals --enable-simd --enable-nontrapping-float-to-int --enable-sign-ext \
        "$dir/${lib}_bg.wasm" -o "$dir/${lib}_bg.wasm"
    du -h "$dir/${lib}_bg.wasm" | sed 's/^/   /'
}

compress() {
    find "$dist" \( -name '*.wasm' -o -name '*.js' \) | while read -r f; do brotli -f -q 11 "$f" -o "$f.br"; done
}

mkdir -p "$dist"
cp "$here/index.html" "$here/loader.js" "$here/isolate-sw.js" "$dist/"
# A page on its own says so; `dspk serve` answers this address with its name instead
echo '{"server": null}' > "$dist/dspk-serve.json"

if [ "$WHICH" = all ] || [ "$WHICH" = mt ]; then
    # Shared memory and atomics, and a standard library built with them (-Z build-std); the
    # exports wasm-bindgen's thread support needs. No compute feature: wgpu cannot cross threads
    export CARGO_TARGET_DIR="$target/web-mt"
    RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128' \
    build mt dspk_app_web rustup run "$TOOLCHAIN" cargo rustc -p dspk-app-web --lib --no-default-features --target wasm32-unknown-unknown --release \
        -Z build-std=std,panic_abort -- \
        -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=4294967296 \
        -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base
fi
if [ "$WHICH" = all ] || [ "$WHICH" = engine ]; then
    # Single thread (a WebGPU device belongs to one thread), wgpu only
    export CARGO_TARGET_DIR="$target/web-engine"
    RUSTFLAGS='-C target-feature=+simd128' \
    build engine dspk_engine_web rustup run "$TOOLCHAIN" cargo build -p dspk-engine-web --lib --target wasm32-unknown-unknown --release
    cp "$root/engine/dspk-engine-web/engine-worker.js" "$root/engine/dspk-engine-web/overview-helper.js" "$root/engine/dspk-engine-web/overview-fold.js" "$dist/engine/"
fi
compress
echo "Built into $dist (serve it: dspk serve --web $dist)"
