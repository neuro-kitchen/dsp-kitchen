//! Plane geometry on data coordinates.

/// Whether `(x, y)` lies inside the closed polygon `vertices` (even-odd rule; fewer than three
/// vertices contain nothing).
pub fn point_in_polygon(x: f64, y: f64, vertices: &[(f64, f64)]) -> bool {
    if vertices.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = vertices.len() - 1;
    for i in 0..vertices.len() {
        let ((xi, yi), (xj, yj)) = (vertices[i], vertices[j]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_in_polygon() {
        let square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        assert!(point_in_polygon(1.0, 1.0, &square));
        assert!(!point_in_polygon(3.0, 1.0, &square));
        // Concave: an L shape leaves its notch outside
        let l = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)];
        assert!(point_in_polygon(0.5, 1.5, &l));
        assert!(!point_in_polygon(1.5, 1.5, &l));
        assert!(!point_in_polygon(0.5, 0.5, &square[..2]));
    }
}
