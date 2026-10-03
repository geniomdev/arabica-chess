use crate::search::{MATE_BOUND, is_mate_score};
use crate::types::Move;

pub const DEFAULT_HASH_MB: usize = 16;
pub const MIN_HASH_MB: usize = 1;
pub const MAX_HASH_MB: usize = 4096;

const BUCKET_SIZE: usize = 4;
const BOUND_BITS: u8 = 2;
const BOUND_MASK: u8 = (1 << BOUND_BITS) - 1;
const GENERATION_CYCLE: u8 = 1 << (u8::BITS as u8 - BOUND_BITS);
const AGE_PENALTY: i32 = 8;
const SAME_POSITION_DEPTH_MARGIN: u8 = 4;
const MISSING_EVAL: i16 = i16::MIN;
const HASHFULL_SAMPLE: usize = 1000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Bound {
    #[default]
    Exact,
    Lower,
    Upper,
}

impl Bound {
    const fn code(self) -> u8 {
        match self {
            Self::Exact => 1,
            Self::Lower => 2,
            Self::Upper => 3,
        }
    }

    const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Exact),
            2 => Some(Self::Lower),
            3 => Some(Self::Upper),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    key: u64,
    best_move: Move,
    score: i16,
    static_eval: i16,
    depth: u8,
    generation_and_bound: u8,
}

impl Entry {
    fn bound(self) -> Option<Bound> {
        Bound::from_code(self.generation_and_bound & BOUND_MASK)
    }

    fn generation(self) -> u8 {
        self.generation_and_bound >> BOUND_BITS
    }

    fn holds(self, key: u64) -> bool {
        self.bound().is_some() && self.key == key
    }

    fn age(self, generation: u8) -> u8 {
        generation.wrapping_sub(self.generation()) % GENERATION_CYCLE
    }

    fn replacement_priority(self, generation: u8) -> i32 {
        if self.bound().is_none() {
            return i32::MIN;
        }
        i32::from(self.depth) - AGE_PENALTY * i32::from(self.age(generation))
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(align(64))]
struct Bucket([Entry; BUCKET_SIZE]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    pub best_move: Option<Move>,
    pub static_eval: Option<i32>,
    score: i32,
    depth: u8,
    bound: Bound,
}

impl Hit {
    pub fn cutoff_score(&self, depth: u8, alpha: i32, beta: i32) -> Option<i32> {
        if self.depth < depth {
            return None;
        }
        let usable = match self.bound {
            Bound::Exact => true,
            Bound::Lower => self.score >= beta,
            Bound::Upper => self.score <= alpha,
        };
        usable.then_some(self.score)
    }

    pub fn refine(&self, static_eval: i32) -> i32 {
        if is_mate_score(self.score) {
            return static_eval;
        }
        let tighter = match self.bound {
            Bound::Exact => true,
            Bound::Lower => self.score > static_eval,
            Bound::Upper => self.score < static_eval,
        };
        if tighter { self.score } else { static_eval }
    }
}

pub struct Store {
    pub key: u64,
    pub best_move: Move,
    pub score: i32,
    pub static_eval: Option<i32>,
    pub depth: u8,
    pub bound: Bound,
    pub ply: usize,
}

#[derive(Default)]
pub struct TranspositionTable {
    buckets: Vec<Bucket>,
    generation: u8,
}

impl TranspositionTable {
    pub fn new(megabytes: usize) -> Self {
        let bytes = megabytes.clamp(MIN_HASH_MB, MAX_HASH_MB) << 20;
        let capacity = bytes / size_of::<Bucket>();
        Self {
            buckets: vec![Bucket::default(); capacity],
            generation: 0,
        }
    }

    pub fn clear(&mut self) {
        self.buckets.fill(Bucket::default());
        self.generation = 0;
    }

    pub fn new_search(&mut self) {
        self.generation = (self.generation + 1) % GENERATION_CYCLE;
    }

    pub fn probe(&self, key: u64, ply: usize) -> Option<Hit> {
        let bucket = self.buckets.get(self.index(key))?;
        let entry = bucket.0.iter().find(|entry| entry.holds(key))?;
        Some(Hit {
            best_move: (entry.best_move != Move::NULL).then_some(entry.best_move),
            static_eval: (entry.static_eval != MISSING_EVAL).then_some(entry.static_eval.into()),
            score: score_from_table(entry.score.into(), ply),
            depth: entry.depth,
            bound: entry.bound()?,
        })
    }

    pub fn store(&mut self, store: Store) {
        let index = self.index(store.key);
        let generation = self.generation;
        let Some(bucket) = self.buckets.get_mut(index) else {
            return;
        };
        let slot = bucket
            .0
            .iter()
            .position(|entry| entry.holds(store.key))
            .or_else(|| {
                (0..BUCKET_SIZE).min_by_key(|&slot| bucket.0[slot].replacement_priority(generation))
            })
            .expect("bucket has slots");
        let entry = &mut bucket.0[slot];
        let same_position = entry.holds(store.key);
        let keeps_deeper_entry = same_position
            && store.bound != Bound::Exact
            && entry.age(generation) == 0
            && store.depth.saturating_add(SAME_POSITION_DEPTH_MARGIN) <= entry.depth;
        if keeps_deeper_entry {
            return;
        }
        let best_move = if store.best_move == Move::NULL && same_position {
            entry.best_move
        } else {
            store.best_move
        };
        *entry = Entry {
            key: store.key,
            best_move,
            score: score_to_table(store.score, store.ply),
            static_eval: store
                .static_eval
                .map_or(MISSING_EVAL, |static_eval| static_eval as i16),
            depth: store.depth,
            generation_and_bound: generation << BOUND_BITS | store.bound.code(),
        };
    }

    pub fn hashfull(&self) -> usize {
        let sample: Vec<Entry> = self
            .buckets
            .iter()
            .flat_map(|bucket| bucket.0)
            .take(HASHFULL_SAMPLE)
            .collect();
        if sample.is_empty() {
            return 0;
        }
        let current = sample
            .iter()
            .filter(|entry| entry.bound().is_some() && entry.age(self.generation) == 0)
            .count();
        current * 1000 / sample.len()
    }

    fn index(&self, key: u64) -> usize {
        ((u128::from(key) * self.buckets.len() as u128) >> 64) as usize
    }
}

fn score_to_table(score: i32, ply: usize) -> i16 {
    let ply = ply as i32;
    let adjusted = if score >= MATE_BOUND {
        score + ply
    } else if score <= -MATE_BOUND {
        score - ply
    } else {
        score
    };
    adjusted as i16
}

fn score_from_table(score: i32, ply: usize) -> i32 {
    let ply = ply as i32;
    if score >= MATE_BOUND {
        score - ply
    } else if score <= -MATE_BOUND {
        score + ply
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::MATE;
    use crate::types::{MoveKind, Square};

    const SAMPLE_KEY: u64 = 0x9E37_79B9_7F4A_7C15;

    fn sample_move() -> Move {
        Move::new(Square::E2, Square::E4, MoveKind::DoublePush)
    }

    fn entry(key: u64, depth: u8, bound: Bound) -> Store {
        Store {
            key,
            best_move: sample_move(),
            score: 0,
            static_eval: Some(0),
            depth,
            bound,
            ply: 0,
        }
    }

    fn same_bucket_keys(table: &TranspositionTable, count: usize) -> Vec<u64> {
        let target = table.index(SAMPLE_KEY);
        (0u64..)
            .map(|step| SAMPLE_KEY.wrapping_add(step.wrapping_mul(0x1_0000_0001)))
            .filter(|&key| table.index(key) == target)
            .take(count)
            .collect()
    }

    #[test]
    fn stores_and_probes_entries() {
        let cases = [
            (10, 3, Bound::Exact, 5, 5, 3, Some(10)),
            (MATE - 5, 2, Bound::Exact, 2, 6, 2, Some(MATE - 9)),
            (-MATE + 7, 4, Bound::Exact, 4, 1, 1, Some(-MATE + 4)),
            (50, 3, Bound::Lower, 0, 0, 4, None),
            (50, 3, Bound::Lower, 0, 0, 3, Some(50)),
            (-50, 3, Bound::Upper, 0, 0, 2, Some(-50)),
            (-50, 0, Bound::Upper, 0, 0, 0, Some(-50)),
        ];

        for (score, depth, bound, store_ply, probe_ply, probe_depth, expected) in cases {
            let mut table = TranspositionTable::new(1);
            table.store(Store {
                score,
                ply: store_ply,
                static_eval: Some(-17),
                ..entry(SAMPLE_KEY, depth, bound)
            });
            let hit = table
                .probe(SAMPLE_KEY, probe_ply)
                .unwrap_or_else(|| panic!("score {score}: entry missing"));
            assert_eq!(hit.best_move, Some(sample_move()), "score {score}");
            assert_eq!(hit.static_eval, Some(-17), "score {score}");
            assert_eq!(
                hit.cutoff_score(probe_depth, -10, 10),
                expected,
                "score {score}, bound {bound:?}, probe depth {probe_depth}"
            );
            assert_eq!(
                table.probe(SAMPLE_KEY ^ 1, probe_ply),
                None,
                "score {score}"
            );
        }
    }

    #[test]
    fn refines_static_eval_with_bounded_score() {
        let cases = [
            (Bound::Exact, 40, 10, 40),
            (Bound::Lower, 40, 10, 40),
            (Bound::Lower, 5, 10, 10),
            (Bound::Upper, 5, 10, 5),
            (Bound::Upper, 40, 10, 10),
            (Bound::Exact, MATE - 3, 10, 10),
        ];

        for (bound, score, static_eval, expected) in cases {
            let mut table = TranspositionTable::new(1);
            table.store(Store {
                score,
                ..entry(SAMPLE_KEY, 3, bound)
            });
            let hit = table.probe(SAMPLE_KEY, 0).expect("entry stored");
            assert_eq!(
                hit.refine(static_eval),
                expected,
                "{bound:?} score {score}, eval {static_eval}"
            );
        }
    }

    #[test]
    fn replaces_shallowest_and_oldest_entries_in_bucket() {
        type DepthAndAge = (u8, u8);
        let cases: [(&[DepthAndAge], Option<usize>); 4] = [
            (&[(5, 0), (9, 0), (7, 0), (6, 0)], Some(0)),
            (&[(5, 0), (9, 0), (7, 2), (6, 0)], Some(2)),
            (&[(20, 1), (9, 0), (7, 0), (6, 0)], Some(3)),
            (&[(5, 0), (9, 0), (7, 0)], None),
        ];

        for (depths_and_ages, expected_evicted) in cases {
            let mut table = TranspositionTable::new(1);
            let keys = same_bucket_keys(&table, depths_and_ages.len() + 1);
            let oldest = depths_and_ages
                .iter()
                .map(|&(_, age)| age)
                .max()
                .unwrap_or(0);
            for generation in 0..=oldest {
                for (&key, &(depth, age)) in keys.iter().zip(depths_and_ages) {
                    if oldest - age == generation {
                        table.store(entry(key, depth, Bound::Exact));
                    }
                }
                if generation < oldest {
                    table.new_search();
                }
            }
            let newcomer = keys[depths_and_ages.len()];
            table.store(entry(newcomer, 1, Bound::Exact));
            let surviving: Vec<bool> = keys
                .iter()
                .map(|&key| table.probe(key, 0).is_some())
                .collect();
            let mut expected = vec![true; keys.len()];
            if let Some(evicted) = expected_evicted {
                expected[evicted] = false;
            }
            assert_eq!(surviving, expected, "entries {depths_and_ages:?}");
        }
    }

    #[test]
    fn keeps_deeper_entry_of_same_position() {
        let cases = [
            (10, Bound::Lower, 0, 5, Bound::Upper, 10),
            (10, Bound::Lower, 0, 7, Bound::Upper, 7),
            (10, Bound::Lower, 0, 2, Bound::Exact, 2),
            (10, Bound::Lower, 1, 2, Bound::Upper, 2),
        ];

        for (old_depth, old_bound, searches_between, new_depth, new_bound, expected_depth) in cases
        {
            let mut table = TranspositionTable::new(1);
            table.store(entry(SAMPLE_KEY, old_depth, old_bound));
            for _ in 0..searches_between {
                table.new_search();
            }
            table.store(Store {
                best_move: Move::NULL,
                ..entry(SAMPLE_KEY, new_depth, new_bound)
            });
            let hit = table.probe(SAMPLE_KEY, 0).expect("entry stored");
            assert_eq!(
                hit.depth, expected_depth,
                "old {old_depth} {old_bound:?}, new {new_depth} {new_bound:?}"
            );
            assert_eq!(hit.best_move, Some(sample_move()), "best move kept");
        }
    }
}
