const UNIT_BITS: u32 = 24;
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

pub const fn splitmix64(value: u64) -> u64 {
    let mut mixed = value.wrapping_add(SPLITMIX_GAMMA);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

pub struct SplitMix(u64);

impl SplitMix {
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub const fn next_u64(&mut self) -> u64 {
        let value = splitmix64(self.0);
        self.0 = self.0.wrapping_add(SPLITMIX_GAMMA);
        value
    }
}

pub struct Random(u64);

impl Random {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> (u64::BITS - UNIT_BITS)) as f32 / (1u32 << UNIT_BITS) as f32
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            items.swap(index, self.below(index + 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffles_into_permutations() {
        let cases = [(1, 0), (2, 1), (3, 2), (4, 17), (5, 1000)];

        for (seed, length) in cases {
            let mut items: Vec<usize> = (0..length).collect();
            Random::new(seed).shuffle(&mut items);
            let mut sorted = items.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..length).collect::<Vec<_>>(), "seed {seed}");
            if length >= 17 {
                assert_ne!(items, sorted, "seed {seed} left {length} items in order");
            }
        }
    }

    #[test]
    fn unit_values_stay_in_range() {
        let seeds = [1, 7, 42, u64::MAX];

        for seed in seeds {
            let mut random = Random::new(seed);
            let values: Vec<f32> = (0..10_000).map(|_| random.unit()).collect();
            assert!(
                values.iter().all(|value| (0.0..1.0).contains(value)),
                "seed {seed}"
            );
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            assert!((mean - 0.5).abs() < 0.02, "seed {seed}, mean {mean}");
        }
    }
}
