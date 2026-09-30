use crate::search::MATE_BOUND;
use crate::types::Move;

pub const DEFAULT_HASH_MB: usize = 16;
pub const MIN_HASH_MB: usize = 1;
pub const MAX_HASH_MB: usize = 4096;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Bound {
    #[default]
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    key: u64,
    best_move: Move,
    score: i16,
    depth: u8,
    bound: Bound,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    pub best_move: Option<Move>,
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
}

pub struct Store {
    pub key: u64,
    pub best_move: Move,
    pub score: i32,
    pub depth: u8,
    pub bound: Bound,
    pub ply: usize,
}

#[derive(Default)]
pub struct TranspositionTable {
    entries: Vec<Entry>,
}

impl TranspositionTable {
    pub fn new(megabytes: usize) -> Self {
        let bytes = megabytes.clamp(MIN_HASH_MB, MAX_HASH_MB) << 20;
        let capacity = bytes / size_of::<Entry>();
        Self {
            entries: vec![Entry::default(); capacity],
        }
    }

    pub fn clear(&mut self) {
        self.entries.fill(Entry::default());
    }

    pub fn probe(&self, key: u64, ply: usize) -> Option<Hit> {
        let entry = self.entries.get(self.index(key))?;
        (entry.key == key && entry.depth > 0).then(|| Hit {
            best_move: (entry.best_move != Move::NULL).then_some(entry.best_move),
            score: score_from_table(entry.score.into(), ply),
            depth: entry.depth,
            bound: entry.bound,
        })
    }

    pub fn store(&mut self, store: Store) {
        let index = self.index(store.key);
        let Some(entry) = self.entries.get_mut(index) else {
            return;
        };
        let same_position = entry.key == store.key;
        let replaceable =
            !same_position || store.depth >= entry.depth || store.bound == Bound::Exact;
        if !replaceable {
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
            depth: store.depth,
            bound: store.bound,
        };
    }

    pub fn hashfull(&self) -> usize {
        let sample = self.entries.len().min(1000);
        if sample == 0 {
            return 0;
        }
        let used = self.entries[..sample]
            .iter()
            .filter(|entry| entry.depth > 0)
            .count();
        used * 1000 / sample
    }

    fn index(&self, key: u64) -> usize {
        ((u128::from(key) * self.entries.len() as u128) >> 64) as usize
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

    #[test]
    fn stores_and_probes_entries() {
        let sample_move = Move::new(Square::E2, Square::E4, MoveKind::DoublePush);
        let cases = [
            (10, 3, Bound::Exact, 5, 5, 3, Some(10)),
            (MATE - 5, 2, Bound::Exact, 2, 6, 2, Some(MATE - 9)),
            (-MATE + 7, 4, Bound::Exact, 4, 1, 1, Some(-MATE + 4)),
            (50, 3, Bound::Lower, 0, 0, 4, None),
            (50, 3, Bound::Lower, 0, 0, 3, Some(50)),
            (-50, 3, Bound::Upper, 0, 0, 2, Some(-50)),
        ];

        for (score, depth, bound, store_ply, probe_ply, probe_depth, expected) in cases {
            let mut table = TranspositionTable::new(1);
            let key = 0x9E37_79B9_7F4A_7C15;
            table.store(Store {
                key,
                best_move: sample_move,
                score,
                depth,
                bound,
                ply: store_ply,
            });
            let hit = table
                .probe(key, probe_ply)
                .unwrap_or_else(|| panic!("score {score}: entry missing"));
            assert_eq!(hit.best_move, Some(sample_move), "score {score}");
            assert_eq!(
                hit.cutoff_score(probe_depth, -10, 10),
                expected,
                "score {score}, bound {bound:?}, probe depth {probe_depth}"
            );
            assert_eq!(table.probe(key ^ 1, probe_ply), None, "score {score}");
        }
    }
}
