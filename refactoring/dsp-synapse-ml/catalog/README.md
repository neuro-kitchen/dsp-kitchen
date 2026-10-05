# Parked catalog entries

`models.json` here is the catalog as it was (commit `HEAD` on 2026-10-05), moved out of
`crates/dsp-synapse-ml/catalog/models.json` because its entries could not be validated: 7 of 10
hashes are hashes of placeholder strings, the Kilosort4 URLs point at files that do not exist,
and the DeepInterpolation DOI is another paper (see `../REVIEW.md`, CAT1–CAT4 and "Provenance
checks"). Only Kilosort4's real `wTEMP.npz` stays in the crate (entry `kilosort4/wtemp-v1`).

Bring an entry back only with a downloaded-and-hashed artifact, a resolved DOI and the upstream
license. Verified paper / code metadata for the parked families is in `../REVIEW.md`.
