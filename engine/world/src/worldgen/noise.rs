//! Feathered-native Perlin gradient noise.
//!
//! Independently implemented classic improved-Perlin: a 256-entry permutation
//! table (seeded via splitmix64 — Feathered's own RNG), gradient sets on the
//! 12 cube edges, quintic fade, trilinear interpolation. An octave stack
//! (`OctaveNoise`) sums N Perlin instances at doubling frequencies with
//! precomputed per-octave input/output factors (the pattern the reference
//! implementations use; the code is Feathered's own).
//!
//! Determinism: a given seed always builds the same permutation tables, and
//! sampling is pure float math — the same input always yields the same
//! output on every platform (no platform `sin`/`random` anywhere).

/// splitmix64 — Feathered's standard deterministic RNG (same family the
/// mesher's hash uses; kept here as the canonical definition).
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One improved-Perlin instance.
#[derive(Debug, Clone)]
pub struct Perlin {
    p: [u8; 512],
    /// Offsets keep lattice coordinates away from exact integers.
    ox: f32,
    oy: f32,
    oz: f32,
}

impl Perlin {
    /// Build from a seed (each seed → a unique permutation table).
    pub fn new(seed: u64) -> Perlin {
        let mut rng = seed;
        // Fisher–Yates over 0..256 using splitmix64 swaps.
        let mut perm = [0u8; 256];
        for (i, v) in perm.iter_mut().enumerate() {
            *v = i as u8;
        }
        for i in (1..256).rev() {
            let j = (splitmix64(&mut rng) % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        let mut p = [0u8; 512];
        for i in 0..512 {
            p[i] = perm[i & 255];
        }
        let f = |rng: &mut u64| (splitmix64(rng) >> 40) as f32 / (1u64 << 24) as f32 * 256.0;
        Perlin {
            p,
            ox: f(&mut rng),
            oy: f(&mut rng),
            oz: f(&mut rng),
        }
    }

    /// Sample in [-1, 1].
    pub fn sample(&self, x: f32, y: f32, z: f32) -> f32 {
        let (x, y, z) = (x + self.ox, y + self.oy, z + self.oz);
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;
        let (tx, ty, tz) = (x - xi as f32, y - yi as f32, z - zi as f32);
        let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let (u, v, w) = (fade(tx), fade(ty), fade(tz));

        // Hash the 8 cube corners through the permutation table. Each
        // stage masks to 8 bits *before* the table lookup (p is 512 entries
        // so the +b/+c adds stay in bounds after masking).
        let h = |a: i32, b: i32, c: i32| -> i32 {
            let i0 = (a & 255) as usize;
            let i1 = (self.p[i0].wrapping_add((b & 255) as u8)) & 255;
            let i2 = (self.p[i1 as usize].wrapping_add((c & 255) as u8)) & 255;
            self.p[i2 as usize] as i32
        };
        let grad = |hash: i32, dx: f32, dy: f32, dz: f32| -> f32 {
            // 12 gradient directions from the cube edges (grad3-style).
            let g = hash & 15;
            let (gx, gy, gz) = match g {
                0..=11 => {
                    let (mut a, mut b) = (
                        1 - (g & 1) * 2, // ±1
                        1 - ((g >> 1) & 1) * 2,
                    );
                    let c = 1 - ((g >> 2) & 1) * 2;
                    if g >= 8 {
                        std::mem::swap(&mut a, &mut b);
                    }
                    (a as f32, b as f32, c as f32)
                }
                12 => (1.0, 1.0, 0.0),
                13 => (-1.0, 1.0, 0.0),
                14 => (1.0, -1.0, 0.0),
                _ => (-1.0, -1.0, 0.0),
            };
            gx * dx + gy * dy + gz * dz
        };

        let aa = h(xi, yi, zi);
        let ba = h(xi + 1, yi, zi);
        let ab = h(xi, yi + 1, zi);
        let bb = h(xi + 1, yi + 1, zi);
        let ac = h(xi, yi, zi + 1);
        let bc = h(xi + 1, yi, zi + 1);
        let ab2 = h(xi, yi + 1, zi + 1);
        let bb2 = h(xi + 1, yi + 1, zi + 1);
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let x1 = lerp(grad(aa, tx, ty, tz), grad(ba, tx - 1.0, ty, tz), u);
        let x2 = lerp(
            grad(ab, tx, ty - 1.0, tz),
            grad(bb, tx - 1.0, ty - 1.0, tz),
            u,
        );
        let y1 = lerp(x1, x2, v);
        let x3 = lerp(
            grad(ac, tx, ty, tz - 1.0),
            grad(bc, tx - 1.0, ty, tz - 1.0),
            u,
        );
        let x4 = lerp(
            grad(ab2, tx, ty - 1.0, tz - 1.0),
            grad(bb2, tx - 1.0, ty - 1.0, tz - 1.0),
            u,
        );
        let y2 = lerp(x3, x4, v);
        lerp(y1, y2, w)
    }
}

/// Octave-stacked Perlin: octaves at doubling frequency, amplitudes from a
/// provided list. Built once per world per field — sampling allocates nothing.
#[derive(Debug, Clone)]
pub struct OctaveNoise {
    octaves: Vec<Perlin>,
    /// input scale per octave (2^i) and amplitude (falling with i).
    factors: Vec<(f32, f32)>,
    /// Normalization bound so output can be mapped to [0, 1) predictably.
    max_value: f32,
}

impl OctaveNoise {
    /// `first_octave` (usually negative = coarse) + amplitudes per octave.
    pub fn new(seed: u64, first_octave: i32, amplitudes: &[f32]) -> OctaveNoise {
        let mut octaves = Vec::with_capacity(amplitudes.len());
        let mut factors = Vec::with_capacity(amplitudes.len());
        let mut max = 0.0f32;
        let mut amp_left = amplitudes.iter().sum::<f32>();
        for (i, &a) in amplitudes.iter().enumerate() {
            if a != 0.0 {
                octaves.push(Perlin::new(seed ^ (0x5EED_0000 + i as u64)));
                // Coarse octaves keep their amplitude; finer ones halve it,
                // and the input scale doubles from the first octave up.
                let freq = 2.0f32.powi(first_octave + i as i32);
                factors.push((freq, a / amp_left.max(1.0) * amp_left));
            }
            amp_left -= a;
            max += a;
        }
        OctaveNoise {
            octaves,
            factors,
            max_value: max.max(1e-6),
        }
    }

    /// Sample in [-1, 1] (normalized by the amplitude bound).
    pub fn sample(&self, x: f32, y: f32, z: f32) -> f32 {
        let mut sum = 0.0f32;
        for (o, (freq, amp)) in self.octaves.iter().zip(&self.factors) {
            sum += o.sample(x * freq, y * freq, z * freq) * amp;
        }
        sum / self.max_value
    }

    /// 2D convenience (y folded to a constant plane).
    pub fn sample2(&self, x: f32, z: f32) -> f32 {
        self.sample(x, 0.1357, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_per_seed() {
        let a = Perlin::new(42).sample(1.7, 2.3, 3.1);
        let b = Perlin::new(42).sample(1.7, 2.3, 3.1);
        assert_eq!(a, b);
        let c = Perlin::new(43).sample(1.7, 2.3, 3.1);
        assert_ne!(a, c);
    }

    #[test]
    fn bounded_output() {
        let n = Perlin::new(7);
        for i in 0..2000 {
            let t = i as f32 * 0.173;
            let v = n.sample(t * 0.31, t * 0.17, t * 0.23);
            assert!(v >= -1.05 && v <= 1.05, "out of range: {v}");
        }
    }

    #[test]
    fn smooth_at_nearby_points() {
        let n = Perlin::new(99);
        let a = n.sample(10.0, 0.0, 10.0);
        let b = n.sample(10.01, 0.0, 10.0);
        assert!(
            (a - b).abs() < 0.1,
            "nearby samples must be close: {a} vs {b}"
        );
    }

    #[test]
    fn octave_noise_bounded_and_deterministic() {
        let a = OctaveNoise::new(5, -4, &[1.0, 1.0, 1.0, 1.0]);
        let b = OctaveNoise::new(5, -4, &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(a.sample2(12.3, -7.7), b.sample2(12.3, -7.7));
        for i in 0..500 {
            let v = a.sample2(i as f32 * 1.31, i as f32 * -0.77);
            assert!(v >= -1.05 && v <= 1.05);
        }
    }
}
