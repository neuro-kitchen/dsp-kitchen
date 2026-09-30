//! Target NWB schema version and the cached specifications written under `/specifications`.

pub const NWB_VERSION: &str = "2.11.0";

/// (namespace, version, [(source name, JSON text)]) for every cached namespace.
pub const SPECS: &[(&str, &str, &[(&str, &str)])] = &[
    (
        "core",
        "2.11.0",
        &[
            ("namespace", include_str!("../../../specs/core/2.11.0/namespace.json")),
            ("nwb.base", include_str!("../../../specs/core/2.11.0/nwb.base.json")),
            ("nwb.behavior", include_str!("../../../specs/core/2.11.0/nwb.behavior.json")),
            ("nwb.device", include_str!("../../../specs/core/2.11.0/nwb.device.json")),
            ("nwb.ecephys", include_str!("../../../specs/core/2.11.0/nwb.ecephys.json")),
            ("nwb.epoch", include_str!("../../../specs/core/2.11.0/nwb.epoch.json")),
            ("nwb.event", include_str!("../../../specs/core/2.11.0/nwb.event.json")),
            ("nwb.file", include_str!("../../../specs/core/2.11.0/nwb.file.json")),
            ("nwb.icephys", include_str!("../../../specs/core/2.11.0/nwb.icephys.json")),
            ("nwb.image", include_str!("../../../specs/core/2.11.0/nwb.image.json")),
            ("nwb.misc", include_str!("../../../specs/core/2.11.0/nwb.misc.json")),
            ("nwb.ogen", include_str!("../../../specs/core/2.11.0/nwb.ogen.json")),
            ("nwb.ophys", include_str!("../../../specs/core/2.11.0/nwb.ophys.json")),
            ("nwb.retinotopy", include_str!("../../../specs/core/2.11.0/nwb.retinotopy.json")),
        ],
    ),
    (
        "hdmf-common",
        "1.10.0",
        &[
            ("base", include_str!("../../../specs/hdmf-common/1.10.0/base.json")),
            ("namespace", include_str!("../../../specs/hdmf-common/1.10.0/namespace.json")),
            ("resources", include_str!("../../../specs/hdmf-common/1.10.0/resources.json")),
            ("sparse", include_str!("../../../specs/hdmf-common/1.10.0/sparse.json")),
            ("table", include_str!("../../../specs/hdmf-common/1.10.0/table.json")),
        ],
    ),
    (
        "hdmf-experimental",
        "0.6.0",
        &[
            ("experimental", include_str!("../../../specs/hdmf-experimental/0.6.0/experimental.json")),
            ("namespace", include_str!("../../../specs/hdmf-experimental/0.6.0/namespace.json")),
        ],
    ),
];
