use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Anchor {
    elo: u32,
    nodes: u64,
    eval_noise: i32,
}

const ANCHORS: [Anchor; 6] = [
    Anchor {
        elo: 600,
        nodes: 64,
        eval_noise: 400,
    },
    Anchor {
        elo: 1000,
        nodes: 200,
        eval_noise: 250,
    },
    Anchor {
        elo: 1500,
        nodes: 1_000,
        eval_noise: 120,
    },
    Anchor {
        elo: 2000,
        nodes: 5_000,
        eval_noise: 50,
    },
    Anchor {
        elo: 2500,
        nodes: 30_000,
        eval_noise: 15,
    },
    Anchor {
        elo: 3000,
        nodes: 200_000,
        eval_noise: 0,
    },
];

pub const MIN_ELO: u32 = ANCHORS[0].elo;
pub const MAX_ELO: u32 = ANCHORS[ANCHORS.len() - 1].elo;
pub const DEFAULT_ELO: u32 = 1500;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handicap {
    pub nodes: u64,
    pub eval_noise: i32,
}

impl Handicap {
    pub fn from_elo(elo: u32) -> Self {
        let elo = elo.clamp(MIN_ELO, MAX_ELO);
        let (lower, upper) = ANCHORS
            .windows(2)
            .map(|pair| (pair[0], pair[1]))
            .find(|(_, upper)| elo <= upper.elo)
            .expect("anchors cover the whole Elo range");
        let fraction = f64::from(elo - lower.elo) / f64::from(upper.elo - lower.elo);
        let node_ratio = upper.nodes as f64 / lower.nodes as f64;
        let nodes = lower.nodes as f64 * node_ratio.powf(fraction);
        let noise_span = f64::from(upper.eval_noise - lower.eval_noise);
        let eval_noise = f64::from(lower.eval_noise) + noise_span * fraction;
        Self {
            nodes: nodes.round() as u64,
            eval_noise: eval_noise.round() as i32,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvalNoise {
    pub amplitude: i32,
    pub seed: u64,
}

impl EvalNoise {
    pub fn offset(self, key: u64) -> i32 {
        if self.amplitude <= 0 {
            return 0;
        }
        let span = 2 * u64::from(self.amplitude.unsigned_abs()) + 1;
        (splitmix64(key ^ self.seed) % span) as i32 - self.amplitude
    }
}

pub fn fresh_noise_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    splitmix64(nanos)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_elo_to_handicap() {
        let cases = [
            (100, 64, 400),
            (600, 64, 400),
            (800, 113, 325),
            (1500, 1_000, 120),
            (1750, 2_236, 85),
            (3000, 200_000, 0),
            (9999, 200_000, 0),
        ];

        for (elo, nodes, eval_noise) in cases {
            assert_eq!(
                Handicap::from_elo(elo),
                Handicap { nodes, eval_noise },
                "elo {elo}"
            );
        }
    }

    #[test]
    fn weakens_monotonically_with_lower_elo() {
        let mut previous = Handicap::from_elo(MIN_ELO);
        for elo in MIN_ELO + 1..=MAX_ELO {
            let current = Handicap::from_elo(elo);
            assert!(current.nodes >= previous.nodes, "nodes at elo {elo}");
            assert!(
                current.eval_noise <= previous.eval_noise,
                "noise at elo {elo}"
            );
            previous = current;
        }
    }

    #[test]
    fn eval_noise_is_deterministic_and_bounded() {
        let cases = [(0, 7), (1, 7), (50, 0), (400, 0xDEAD_BEEF)];

        for (amplitude, seed) in cases {
            let noise = EvalNoise { amplitude, seed };
            for key in (0..1_000u64).map(splitmix64) {
                let offset = noise.offset(key);
                assert!(
                    offset.abs() <= amplitude,
                    "amplitude {amplitude}, key {key}"
                );
                assert_eq!(
                    offset,
                    noise.offset(key),
                    "amplitude {amplitude}, key {key}"
                );
            }
        }
    }
}
