//! Voxel raycasting (block targeting).
//!
//! Amanatides–Woo style DDA: walk the voxel grid along the ray, one boundary
//! crossing at a time. Returns the first targetable voxel plus the face the
//! ray entered through, which is exactly what block placement needs. Pure
//! CPU, allocation-free, and exact — no fixed-step marching.

/// Which face of the voxel the ray entered through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    /// Ray entered moving +x (hit the voxel's −x face).
    West,
    /// Entered moving −x (hit the +x face).
    East,
    /// Entered moving +y (hit the bottom face).
    Down,
    /// Entered moving −y (hit the top face).
    Up,
    /// Entered moving +z (hit the −z face).
    North,
    /// Entered moving −z (hit the +z face).
    South,
}

impl Face {
    /// Outward normal of the face.
    pub fn normal(self) -> [f32; 3] {
        match self {
            Face::West => [-1.0, 0.0, 0.0],
            Face::East => [1.0, 0.0, 0.0],
            Face::Down => [0.0, -1.0, 0.0],
            Face::Up => [0.0, 1.0, 0.0],
            Face::North => [0.0, 0.0, -1.0],
            Face::South => [0.0, 0.0, 1.0],
        }
    }

    /// Voxel the face opens onto (the placement position).
    pub fn neighbor(self, x: i64, y: i64, z: i64) -> (i64, i64, i64) {
        let n = self.normal();
        (x + n[0] as i64, y + n[1] as i64, z + n[2] as i64)
    }
}

/// A hit: the target voxel, the entered face, and the exact entry point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub x: i64,
    pub y: i64,
    pub z: i64,
    pub face: Face,
    /// World-space point where the ray entered the voxel.
    pub point: [f32; 3],
    /// Distance from the ray origin to `point` (blocks).
    pub distance: f32,
}

/// Predicate deciding whether a voxel stops the ray. The client wires this
/// to "the block is targetable" (anything not air); tests pass closures.
pub type HitsAt<'a> = dyn Fn(i64, i64, i64) -> bool + 'a;

/// Cast a ray from `origin` along `dir` (need not be normalized), up to
/// `max_dist`. Returns the first voxel for which `hits` is true, excluding
/// the voxel containing the origin (the eye voxel is never targeted —
/// vanilla behavior — so players inside foliage can still look outward).
pub fn raycast(origin: [f32; 3], dir: [f32; 3], max_dist: f32, hits: &HitsAt) -> Option<RayHit> {
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if len < f32::EPSILON {
        return None;
    }
    let d = [dir[0] / len, dir[1] / len, dir[2] / len];

    // Current voxel.
    let mut voxel = [
        origin[0].floor() as i64,
        origin[1].floor() as i64,
        origin[2].floor() as i64,
    ];

    // Step direction per axis. NB: f32::signum(0.0) is 1.0, so a plain
    // signum would make zero-length axes step forever — map 0 to 0.
    let sgn = |v: f32| -> i64 {
        if v > 0.0 {
            1
        } else if v < 0.0 {
            -1
        } else {
            0
        }
    };
    let step = [sgn(d[0]), sgn(d[1]), sgn(d[2])];

    // t-delta: ray length to cross one full voxel per axis.
    let t_delta = [
        if d[0] != 0.0 { 1.0 / d[0].abs() } else { f32::INFINITY },
        if d[1] != 0.0 { 1.0 / d[1].abs() } else { f32::INFINITY },
        if d[2] != 0.0 { 1.0 / d[2].abs() } else { f32::INFINITY },
    ];

    // t at which the ray reaches each axis's next boundary (ray-parameter
    // t: origin + d·t; d is normalized so t = world distance).
    let t_next = |o: f32, i: i64, s: i64, d_axis: f32| -> f32 {
        if s > 0 {
            ((i + 1) as f32 - o) / d_axis
        } else if s < 0 {
            (o - i as f32) / d_axis
        } else {
            f32::INFINITY
        }
    };
    let mut t_max = [
        t_next(origin[0], voxel[0], step[0], d[0]),
        t_next(origin[1], voxel[1], step[1], d[1]),
        t_next(origin[2], voxel[2], step[2], d[2]),
    ];

    // The face the ray entered the current voxel through (None for the
    // origin voxel, which is skipped).
    let mut entered_from: Option<Face> = None;
    let mut entry_t: f32 = 0.0;

    loop {
        if let Some(face) = entered_from {
            if hits(voxel[0], voxel[1], voxel[2]) {
                return Some(RayHit {
                    x: voxel[0],
                    y: voxel[1],
                    z: voxel[2],
                    face,
                    point: [
                        origin[0] + d[0] * entry_t,
                        origin[1] + d[1] * entry_t,
                        origin[2] + d[2] * entry_t,
                    ],
                    distance: entry_t,
                });
            }
        }

        // Advance across the nearest boundary.
        let axis = if t_max[0] < t_max[1] {
            if t_max[0] < t_max[2] {
                0
            } else {
                2
            }
        } else if t_max[1] < t_max[2] {
            1
        } else {
            2
        };
        let t = t_max[axis];
        if t > max_dist {
            return None;
        }
        voxel[axis] += step[axis];
        t_max[axis] += t_delta[axis];
        entry_t = t;
        entered_from = Some(match (axis, step[axis] > 0) {
            (0, true) => Face::West,
            (0, false) => Face::East,
            (1, true) => Face::Down,
            (1, false) => Face::Up,
            (2, true) => Face::North,
            (_, _) => Face::South,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_ray_down_hits_the_top_face() {
        // Floor: solid for y <= 9 (top face at y = 10).
        let hits = |_: i64, y: i64, _: i64| y <= 9;
        let hit = raycast([0.5, 20.0, 0.5], [0.0, -1.0, 0.0], 32.0, &hits).expect("hit floor");
        assert_eq!((hit.x, hit.y, hit.z), (0, 9, 0));
        assert_eq!(hit.face, Face::Up);
        assert!((hit.distance - 10.0).abs() < 1e-5, "eye is 10 above the face");
        assert!((hit.point[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn horizontal_ray_hits_a_west_facing_wall() {
        // Wall column at x = 5.
        let hits = |x: i64, _: i64, _: i64| x == 5;
        let hit = raycast([1.5, 10.0, 0.5], [1.0, 0.0, 0.0], 16.0, &hits).expect("hit wall");
        assert_eq!((hit.x, hit.y, hit.z), (5, 10, 0));
        assert_eq!(hit.face, Face::West, "ray moving +x enters the −x face");
        assert!((hit.point[0] - 5.0).abs() < 1e-5);
        assert!((hit.distance - 3.5).abs() < 1e-5);
    }

    #[test]
    fn placement_neighbor_is_adjacent_to_the_hit_face() {
        let hits = |x: i64, _: i64, _: i64| x == 5;
        let hit = raycast([1.5, 10.0, 0.5], [1.0, 0.0, 0.0], 16.0, &hits).unwrap();
        assert_eq!(hit.face.neighbor(hit.x, hit.y, hit.z), (4, 10, 0));
        // Top face: neighbor above.
        let down = raycast([0.5, 20.0, 0.5], [0.0, -1.0, 0.0], 32.0, &|_: i64, y: i64, _: i64| y <= 9).unwrap();
        assert_eq!(down.face.neighbor(down.x, down.y, down.z), (0, 10, 0));
    }

    #[test]
    fn diagonal_ray_hits_the_expected_voxel() {
        // Single solid voxel at (3, 10, 3).
        let hits = |x: i64, y: i64, z: i64| (x, y, z) == (3, 10, 3);
        let hit = raycast(
            [0.5, 10.5, 0.5],
            [3.0, 0.0, 3.0],
            16.0,
            &hits,
        )
        .expect("diagonal must hit the cube");
        assert_eq!((hit.x, hit.y, hit.z), (3, 10, 3));
        // Moving +x and +z equally: the ray enters exactly at the corner
        // (x=3, z=3); the tie must resolve to a definite face whose normal
        // is one of the two axes.
        assert!(
            hit.face == Face::West || hit.face == Face::North,
            "corner entry must be x or z, got {:?}",
            hit.face
        );
        assert!(
            (hit.point[0] - 3.0).abs() < 1e-5 || (hit.point[2] - 3.0).abs() < 1e-5,
            "entry point on the voxel face, got {:?}",
            hit.point
        );
    }

    #[test]
    fn eye_voxel_is_never_targeted() {
        // Eye standing inside a solid voxel.
        let hits = |_: i64, _: i64, _: i64| true;
        let hit = raycast([2.5, 10.5, 2.5], [0.0, -1.0, 0.0], 8.0, &hits);
        assert!(
            hit.map(|h| (h.x, h.y, h.z) != (2, 10, 2)).unwrap_or(true),
            "the origin voxel must be skipped"
        );
        // Straight down: the next voxel below is hit.
        let hit = raycast([2.5, 10.5, 2.5], [0.0, -1.0, 0.0], 8.0, &hits).unwrap();
        assert_eq!((hit.x, hit.y, hit.z), (2, 9, 2));
        assert_eq!(hit.face, Face::Up);
    }

    #[test]
    fn max_distance_limits_the_reach() {
        let hits = |x: i64, _: i64, _: i64| x == 5;
        assert!(raycast([1.5, 10.0, 0.5], [1.0, 0.0, 0.0], 3.0, &hits).is_none());
        assert!(raycast([1.5, 10.0, 0.5], [1.0, 0.0, 0.0], 3.6, &hits).is_some());
    }

    #[test]
    fn miss_returns_none() {
        let hits = |_: i64, _: i64, _: i64| false;
        assert!(raycast([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 64.0, &hits).is_none());
        // Zero direction is a miss, not a crash.
        assert!(raycast([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 64.0, &hits).is_none());
    }

    #[test]
    fn upward_ray_hits_a_bottom_face() {
        let hits = |_: i64, y: i64, _: i64| y == 12;
        let hit = raycast([1.5, 10.0, 1.5], [0.0, 1.0, 0.0], 16.0, &hits).unwrap();
        assert_eq!((hit.x, hit.y, hit.z), (1, 12, 1));
        assert_eq!(hit.face, Face::Down, "ray moving +y enters the bottom face");
    }

    #[test]
    fn negative_directions_report_flipped_faces() {
        let hits = |x: i64, _: i64, _: i64| x == -4;
        let hit = raycast([0.5, 10.0, 0.5], [-1.0, 0.0, 0.0], 16.0, &hits).unwrap();
        assert_eq!((hit.x, hit.y, hit.z), (-4, 10, 0));
        assert_eq!(hit.face, Face::East, "ray moving −x enters the +x face");
    }
}
