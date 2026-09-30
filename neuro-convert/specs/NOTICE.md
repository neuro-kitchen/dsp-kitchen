# Vendored schema files

Cached copies of the NWB and HDMF schemas, written into every NWB file under `/specifications`
(as pynwb / hdmf-zarr do) so readers do not depend on locally installed schema versions.

| Namespace | Version | Source | License |
|---|---|---|---|
| core (NWB) | 2.11.0 | https://github.com/NeurodataWithoutBorders/nwb-schema | BSD-3-Clause-LBNL |
| hdmf-common | 1.10.0 | https://github.com/hdmf-dev/hdmf-common-schema | BSD-3-Clause-LBNL |
| hdmf-experimental | 0.6.0 | https://github.com/hdmf-dev/hdmf-common-schema | BSD-3-Clause-LBNL |

Extracted from a file written by pynwb 4.2.0 / hdmf 6.2.0 / hdmf-zarr 0.14.0 (the JSON text
stored for each schema source).
