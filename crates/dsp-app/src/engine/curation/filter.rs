//! Phy-style cluster filter expression evaluator (`group == 'good' && fr > 1`, `ch >= 10`, or
//! plain substring matching).

use super::ClusterMeta;

/// Returns `true` if `cluster` matches `query`.
///
/// Supports:
/// - Empty query (matches everything)
/// - Plain integer / substring (matches cluster id, group, KS label, or channel)
/// - Boolean expressions with `&&` / `and` and `||` / `or` over comparisons (`==`, `!=`, `>=`,
///   `<=`, `>`, `<`) on fields: `id`, `ch` / `channel`, `depth`, `sh` / `shank`, `n_spikes` /
///   `spikes`, `fr` / `firing_rate`, `amp` / `amplitude`, `contam` / `contam_pct`, `ks_label`,
///   `group`, or any custom label key.
pub fn matches_filter(query: &str, cluster: &ClusterMeta) -> bool {
    let q = query.trim();
    if q.is_empty() {
        return true;
    }
    let has_op = ["==", "!=", ">=", "<=", ">", "<"].iter().any(|op| q.contains(op));
    if !has_op {
        let low = q.to_lowercase();
        return cluster.id.to_string().contains(&low)
            || cluster.group.as_str().contains(&low)
            || cluster.ks_label.to_lowercase().contains(&low)
            || format!("ch{}", cluster.ch).contains(&low)
            || cluster.custom.values().any(|v| v.to_lowercase().contains(&low));
    }

    // Split on `||` or ` or `
    let normalized = q.replace(" and ", " && ").replace(" AND ", " && ").replace(" or ", " || ").replace(" OR ", " || ");
    normalized.split("||").any(|or_part| {
        or_part.split("&&").all(|clause| eval_clause(clause.trim(), cluster))
    })
}

fn eval_clause(clause: &str, c: &ClusterMeta) -> bool {
    if clause.is_empty() {
        return true;
    }
    for op in ["==", "!=", ">=", "<=", ">", "<"] {
        if let Some((lhs, rhs)) = clause.split_once(op) {
            let field = lhs.trim().to_lowercase();
            let raw_rhs = rhs.trim().trim_matches(|ch| ch == '\'' || ch == '"');
            if let Some(num_lhs) = numeric_field(&field, c) {
                if let Ok(num_rhs) = raw_rhs.parse::<f64>() {
                    return match op {
                        "==" => (num_lhs - num_rhs).abs() < 1e-6,
                        "!=" => (num_lhs - num_rhs).abs() >= 1e-6,
                        ">=" => num_lhs >= num_rhs,
                        "<=" => num_lhs <= num_rhs,
                        ">" => num_lhs > num_rhs,
                        "<" => num_lhs < num_rhs,
                        _ => false,
                    };
                }
            }
            if let Some(str_lhs) = string_field(&field, c) {
                let a = str_lhs.to_lowercase();
                let b = raw_rhs.to_lowercase();
                return match op {
                    "==" => a == b,
                    "!=" => a != b,
                    _ => false,
                };
            }
            return false;
        }
    }
    false
}

fn numeric_field(field: &str, c: &ClusterMeta) -> Option<f64> {
    match field {
        "id" | "cluster" | "cluster_id" => Some(c.id as f64),
        "ch" | "chan" | "channel" => Some(c.ch as f64),
        "depth" | "y" => Some(c.depth as f64),
        "sh" | "shank" => Some(c.sh as f64),
        "n_spikes" | "spikes" | "n" | "count" => Some(c.n_spikes as f64),
        "fr" | "firing_rate" | "rate" => Some(c.fr as f64),
        "amp" | "amplitude" => Some(c.amp as f64),
        "contam" | "contam_pct" | "isi_viol" => Some(c.contam_pct as f64),
        other => c.custom.get(other).and_then(|v| v.parse::<f64>().ok()),
    }
}

fn string_field<'a>(field: &str, c: &'a ClusterMeta) -> Option<&'a str> {
    match field {
        "group" | "label" => Some(c.group.as_str()),
        "ks_label" | "kslabel" => Some(&c.ks_label),
        other => c.custom.get(other).map(String::as_str),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::curation::ClusterGroup;
    use std::collections::BTreeMap;

    fn sample_cluster() -> ClusterMeta {
        let mut custom = BTreeMap::new();
        custom.insert("region".into(), "CA1".into());
        ClusterMeta {
            id: 7,
            ch: 14,
            depth: 280.0,
            sh: 0,
            n_spikes: 1250,
            fr: 12.5,
            amp: 48.2,
            contam_pct: 1.4,
            ks_label: "good".into(),
            group: ClusterGroup::Good,
            custom,
        }
    }

    #[test]
    fn test_matches_filter_expressions() {
        let c = sample_cluster();
        assert!(matches_filter("", &c));
        assert!(matches_filter("7", &c));
        assert!(matches_filter("good", &c));
        assert!(matches_filter("group == 'good' && fr > 5", &c));
        assert!(!matches_filter("group == 'mua' && fr > 5", &c));
        assert!(matches_filter("group == 'mua' || contam < 2.0", &c));
        assert!(matches_filter("region == 'ca1' && n_spikes >= 1000", &c));
    }
}
