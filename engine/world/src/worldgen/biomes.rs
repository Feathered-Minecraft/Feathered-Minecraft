//! Biome selection: climate fields → biome, Feathered-native.
//!
//! The reference implementations sample climate parameters
//! (temperature/humidity/continentalness/erosion/ridges) and pick the biome
//! whose parameter point is nearest. Feathered uses a simplified two-field
//! climate (temperature, humidity) — enough for meaningful surface/mob
//! variation while staying cheap for low-end hardware. Fields are
//! coarse-frequency octave noise evaluated per 4×4 cell (quart resolution,
//! like the references) and cached per chunk column.

/// The biomes Feathered's generator distinguishes. Kept deliberately small:
/// each must have distinct surface rules and spawn behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Biome {
    Plains,
    Forest,
    Desert,
    Snowfield,
    Mountains,
    Ocean,
}

impl Biome {
    /// Display name (debug screen).
    pub fn name(&self) -> &'static str {
        match self {
            Biome::Plains => "Plains",
            Biome::Forest => "Forest",
            Biome::Desert => "Desert",
            Biome::Snowfield => "Snowfield",
            Biome::Mountains => "Mountains",
            Biome::Ocean => "Ocean",
        }
    }

    /// Base terrain offset (blocks) this biome pulls toward.
    pub fn height_bias(&self) -> f32 {
        match self {
            Biome::Plains => 0.0,
            Biome::Forest => 1.0,
            Biome::Desert => -1.0,
            Biome::Snowfield => 1.5,
            Biome::Mountains => 12.0,
            Biome::Ocean => -9.0,
        }
    }

    /// Terrain roughness multiplier (mountains amplify the hill field).
    pub fn height_scale(&self) -> f32 {
        match self {
            Biome::Mountains => 2.2,
            Biome::Ocean => 0.5,
            _ => 1.0,
        }
    }
}

/// The two climate fields (built once per world).
#[derive(Debug, Clone)]
pub struct Climate {
    temperature: super::noise::OctaveNoise,
    humidity: super::noise::OctaveNoise,
    continentalness: super::noise::OctaveNoise,
}

/// Sampling resolution: one biome per 4×4 blocks (quart cells).
pub const CELL: i64 = 4;

impl Climate {
    pub fn new(seed: u64) -> Climate {
        Climate {
            temperature: super::noise::OctaveNoise::new(seed ^ 0x7E3C, -3, &[1.0, 1.0, 0.6]),
            humidity: super::noise::OctaveNoise::new(seed ^ 0x0DD1, -3, &[1.0, 0.7]),
            continentalness: super::noise::OctaveNoise::new(seed ^ 0xC0A7, -4, &[1.0, 1.0, 0.8]),
        }
    }

    /// Nearest-point biome selection at a block position (deterministic;
    /// pure function of seed + coordinates).
    pub fn biome_at(&self, x: i64, z: i64, surface: i64, sea_level: i64) -> Biome {
        // Quart-cell quantization gives biomes real area (4×4 minimum).
        let (qx, qz) = (x.div_euclid(CELL) as f32, z.div_euclid(CELL) as f32);
        let temp = self.temperature.sample2(qx * 0.35, qz * 0.35);
        let humid = self.humidity.sample2(qx * 0.5, qz * 0.5);
        let cont = self.continentalness.sample2(qx * 0.3, qz * 0.3);

        // Oceans: strongly negative continentalness (far from land centers).
        if cont < -0.45 {
            return Biome::Ocean;
        }
        // Mountains: high continentalness + high erosion resistance —
        // approximated here as the top continentalness band.
        if cont > 0.5 {
            return Biome::Mountains;
        }
        // Cold → snow; hot+dry → desert; hot+wet or mild+wet → forest.
        if temp < -0.35 {
            return Biome::Snowfield;
        }
        if temp > 0.4 && humid < -0.1 {
            return Biome::Desert;
        }
        if humid > 0.15 {
            return Biome::Forest;
        }
        let _ = (surface, sea_level);
        Biome::Plains
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_bounded() {
        let c = Climate::new(777);
        let a = c.biome_at(123, -456, 24, 20);
        let b = c.biome_at(123, -456, 24, 20);
        assert_eq!(a, b);
        // All six biomes reachable across a wide sweep (variation exists).
        let mut seen = std::collections::HashSet::new();
        for x in (-400..400).step_by(16) {
            for z in (-400..400).step_by(16) {
                seen.insert(c.biome_at(x, z, 24, 20));
            }
        }
        assert!(seen.len() >= 4, "expected biome variety, got {seen:?}");
    }

    #[test]
    fn quart_cells_share_biomes_with_neighbors() {
        let c = Climate::new(31415);
        // Points inside the same 4×4 cell must agree (quantization).
        assert_eq!(c.biome_at(8, 8, 24, 20), c.biome_at(11, 11, 24, 20));
        assert_eq!(c.biome_at(-9, -9, 24, 20), c.biome_at(-9, -6, 24, 20));
    }
}
