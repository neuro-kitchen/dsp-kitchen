use serde::{Deserialize, Serialize};

/// Physical unit of a channel's scaled values (`stored * gain + offset`).
/// Microvolts in one volt (formats storing volts per bit report µV).
pub const MICROVOLTS_PER_VOLT: f64 = 1e6;

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignalUnit {
    Volt,
    Millivolt,
    Microvolt,
    Ampere,
    Milliampere,
    Microampere,
    /// Normalized values or raw counts.
    #[default]
    Dimensionless,
    /// Any unit not listed, by its symbol (e.g. `"Pa"`, `"g"`).
    Other(String),
}

impl SignalUnit {
    /// Display symbol (`"µV"`, `"mA"`, …); empty for [`SignalUnit::Dimensionless`].
    pub fn symbol(&self) -> &str {
        match self {
            SignalUnit::Volt => "V",
            SignalUnit::Millivolt => "mV",
            SignalUnit::Microvolt => "µV",
            SignalUnit::Ampere => "A",
            SignalUnit::Milliampere => "mA",
            SignalUnit::Microampere => "µA",
            SignalUnit::Dimensionless => "",
            SignalUnit::Other(symbol) => symbol,
        }
    }
}
