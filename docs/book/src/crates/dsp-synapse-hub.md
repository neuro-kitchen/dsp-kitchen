# dsp-synapse-hub

## Intent

Fetch published artifacts (model weights, sorter arrays) once, verify them, and keep them in a
local cache. It knows nothing about what the files mean: catalogs belong to the crates that
publish them ([dsp-synapse-ml](dsp-synapse-ml.md)). It is the only place model code touches the
network.

## Usage

```rust,ignore
use dsp_synapse_hub::{Artifact, Hub};

let hub = Hub::from_env_or_default();          // $DSP_KITCHEN_HUB_DIR or <user cache>/dsp-kitchen/hub
let wtemp = Artifact {
    id: "kilosort4/wtemp-v1".into(),
    family: "kilosort4".into(),
    file_name: "wTEMP.npz".into(),
    url: "https://osf.io/download/6807fb5958b763aae139aa60/".into(),
    sha256: "cae1c96f8f4150be0a39627515750b70c4bc3548177cf487ae3c013f1ca6abd8".into(),
    size_bytes: 3432,
};
let path = hub.pull(&wtemp, false)?;           // downloads only if not installed
```

With dsp-synapse-ml's `hub` feature, `ModelHub` builds the `Artifact` from a catalog entry
(`ModelManifest::artifact()`).

## Reference

| Item | Purpose |
|---|---|
| `Artifact` | id (`family/slug`), family, file name, URI, SHA-256, size. |
| `Hub::pull(artifact, force)` | Download into the cache unless installed; returns the local path. |
| `Hub::status` | `Installed` / `Available` / `Corrupted` from the cache index and the file size (no hashing). |
| `Hub::verify(artifact, check_link)` | Re-hash the cached file (error when corrupted); optionally probe the URL. |
| `Hub::remove`, `Hub::clean` | Remove one artifact / everything. |
| `download_and_verify` | Stream to `<file>.part` while hashing; commit only when the size and SHA-256 match. **An artifact without a hash is refused.** |
| `resolve_weights_uri` | `hf://org/repo[@rev]/path` → huggingface.co `resolve`; `gh://org/repo[@rev]/path` → raw.githubusercontent.com; `zenodo://record/file` → zenodo.org; `https://`, `file://` as is. |

Cache layout: `<root>/models/<family>/<slug>/<file>` and `<root>/index.json` (hash, size and
source of each installed file).

## Limitations

- `gh://` resolves to `raw.githubusercontent.com`, which does not serve Git-LFS files or release
  assets; use `https://` for those.
- Downloads are blocking (`reqwest::blocking`).
