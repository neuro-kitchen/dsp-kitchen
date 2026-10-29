## What and why

<!-- What this changes and why; link the issue it closes (Closes #…). -->

## Checks

- [ ] Built and tested with the one profile: `cargo test --workspace --exclude dspk-app --exclude dspk-py --profile validate --no-run`, then the test binaries (5-minute limit each)
- [ ] Runtimes the tests ran on: <!-- wgpu / CUDA / CPU; CI runs the CPU runtime only -->
- [ ] Python tests, if the bindings or the operations they call changed: `maturin develop --profile validate`, then `pytest bindings/python/tests`
- [ ] Results checked against their reference (scipy / NumPy, upstream sorter) where they can change
- [ ] Docs and the CHANGELOG updated if behaviour or API changed

## Licences

- [ ] No MIT / Apache-licensed crate gains a dependency on `dspk-ks4` (GPL-3.0); code ported from elsewhere keeps its upstream notice (see CONTRIBUTING.md)
