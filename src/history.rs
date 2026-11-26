use crate::params;
use crate::types::{Bitboard, Color, Move, PIECE_TYPES, Piece, SIDES, SQUARES, Square};

pub const CONTINUATION_PLIES: usize = 2;

const MAX_HISTORY: i32 = 16_384;
const PIECE_TO_SLOTS: usize = SIDES * PIECE_TYPES * SQUARES;
const THREAT_BUCKETS: usize = 4;
const VICTIM_SLOTS: usize = PIECE_TYPES + 1;
const CORRECTION_BITS: u32 = 14;
const CORRECTION_SLOTS: usize = 1 << CORRECTION_BITS;
pub const HASHED_CORRECTIONS: usize = 3;
const CORRECTION_GRAIN: i32 = 256;
const CORRECTION_WEIGHT_SCALE: i32 = 256;
const MAX_CORRECTION_WEIGHT: i32 = 16;
const MAX_CORRECTION: i32 = 64 * CORRECTION_GRAIN;
const MAX_CORRECTION_BONUS: i32 = MAX_CORRECTION / 4;

pub type Continuations = [Option<PieceTo>; CONTINUATION_PLIES];

#[derive(Clone, Copy, Debug)]
pub struct CorrectionKeys {
    pub side: Color,
    pub hashed: [u64; HASHED_CORRECTIONS],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PieceTo {
    pub side: Color,
    pub piece: Piece,
    pub to: Square,
}

impl PieceTo {
    fn index(self) -> usize {
        (self.side.index() * PIECE_TYPES + self.piece.index()) * SQUARES + self.to.index()
    }
}

pub struct History {
    butterfly: Vec<i32>,
    continuation: [Vec<i32>; CONTINUATION_PLIES],
    capture: Vec<i32>,
    correction: [Vec<i16>; HASHED_CORRECTIONS],
}

impl Default for History {
    fn default() -> Self {
        Self {
            butterfly: vec![0; SIDES * THREAT_BUCKETS * SQUARES * SQUARES],
            continuation: std::array::from_fn(|_| vec![0; PIECE_TO_SLOTS * PIECE_TO_SLOTS]),
            capture: vec![0; PIECE_TO_SLOTS * VICTIM_SLOTS],
            correction: std::array::from_fn(|_| vec![0; SIDES * CORRECTION_SLOTS]),
        }
    }
}

pub fn history_bonus(depth: i32) -> i32 {
    (params::history_bonus_scale() * depth * depth).min(params::history_bonus_max())
}

fn apply_gravity(entry: &mut i32, bonus: i32) {
    *entry += bonus - *entry * bonus.abs() / MAX_HISTORY;
}

fn butterfly_index(side: Color, quiet: Move, threats: Bitboard) -> usize {
    let from_threatened = usize::from(threats.contains(quiet.from()));
    let to_threatened = usize::from(threats.contains(quiet.to()));
    let bucket = side.index() * THREAT_BUCKETS + from_threatened * 2 + to_threatened;
    (bucket * SQUARES + quiet.from().index()) * SQUARES + quiet.to().index()
}

fn continuation_index(previous: PieceTo, moved: PieceTo) -> usize {
    previous.index() * PIECE_TO_SLOTS + moved.index()
}

fn capture_index(moved: PieceTo, victim: Option<Piece>) -> usize {
    moved.index() * VICTIM_SLOTS + victim.map_or(PIECE_TYPES, Piece::index)
}

fn nudge_correction(entry: &mut i16, bonus: i32) {
    let entry_value = i32::from(*entry);
    let nudged = entry_value + bonus - entry_value * bonus.abs() / MAX_CORRECTION;
    *entry = nudged.clamp(-MAX_CORRECTION, MAX_CORRECTION) as i16;
}

fn correction_index(side: Color, key: u64) -> usize {
    side.index() * CORRECTION_SLOTS + (key >> (u64::BITS - CORRECTION_BITS)) as usize
}

impl History {
    pub fn clear(&mut self) {
        self.butterfly.fill(0);
        for table in &mut self.continuation {
            table.fill(0);
        }
        self.capture.fill(0);
        for table in &mut self.correction {
            table.fill(0);
        }
    }

    pub fn quiet_score(
        &self,
        quiet: Move,
        moved: PieceTo,
        continuations: Continuations,
        threats: Bitboard,
    ) -> i32 {
        let continuation_score: i32 = self
            .continuation
            .iter()
            .zip(continuations)
            .filter_map(|(table, previous)| {
                previous.map(|previous| table[continuation_index(previous, moved)])
            })
            .sum();
        self.butterfly[butterfly_index(moved.side, quiet, threats)] + continuation_score
    }

    pub fn capture_score(&self, moved: PieceTo, victim: Option<Piece>) -> i32 {
        self.capture[capture_index(moved, victim)]
    }

    pub fn update_quiet(
        &mut self,
        quiet: Move,
        moved: PieceTo,
        continuations: Continuations,
        threats: Bitboard,
        bonus: i32,
    ) {
        apply_gravity(
            &mut self.butterfly[butterfly_index(moved.side, quiet, threats)],
            bonus,
        );
        for (table, previous) in self.continuation.iter_mut().zip(continuations) {
            if let Some(previous) = previous {
                apply_gravity(&mut table[continuation_index(previous, moved)], bonus);
            }
        }
    }

    pub fn update_butterfly(&mut self, quiet: Move, side: Color, threats: Bitboard, bonus: i32) {
        apply_gravity(
            &mut self.butterfly[butterfly_index(side, quiet, threats)],
            bonus,
        );
    }

    pub fn update_capture(&mut self, moved: PieceTo, victim: Option<Piece>, bonus: i32) {
        apply_gravity(&mut self.capture[capture_index(moved, victim)], bonus);
    }

    pub fn corrected_eval(&self, keys: &CorrectionKeys, raw_eval: i32) -> i32 {
        let correction: i32 = self
            .correction
            .iter()
            .zip(keys.hashed)
            .map(|(table, key)| i32::from(table[correction_index(keys.side, key)]))
            .sum();
        raw_eval + correction / CORRECTION_GRAIN
    }

    pub fn update_correction(&mut self, keys: &CorrectionKeys, depth: i32, residual: i32) {
        let weight = (depth + 1).min(MAX_CORRECTION_WEIGHT);
        let bonus = (residual * CORRECTION_GRAIN * weight / CORRECTION_WEIGHT_SCALE)
            .clamp(-MAX_CORRECTION_BONUS, MAX_CORRECTION_BONUS);
        for (table, key) in self.correction.iter_mut().zip(keys.hashed) {
            nudge_correction(&mut table[correction_index(keys.side, key)], bonus);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MoveKind;

    const KNIGHT_TO_F3: PieceTo = PieceTo {
        side: Color::White,
        piece: Piece::Knight,
        to: Square::F3,
    };
    const PAWN_TO_E5: PieceTo = PieceTo {
        side: Color::Black,
        piece: Piece::Pawn,
        to: Square::E5,
    };
    const BISHOP_TO_C4: PieceTo = PieceTo {
        side: Color::White,
        piece: Piece::Bishop,
        to: Square::C4,
    };

    #[test]
    fn accumulates_quiet_history_per_context() {
        let knight_move = Move::new(Square::G1, Square::F3, MoveKind::Quiet);
        let full_context = [Some(PAWN_TO_E5), Some(BISHOP_TO_C4)];
        let landing_attacked = Bitboard::from_square(Square::F3);
        let origin_attacked = Bitboard::from_square(Square::G1);
        let unrelated_attacked = Bitboard::from_square(Square::E4);
        let bonus = history_bonus(4);
        let cases = [
            (full_context, landing_attacked, 3 * bonus),
            ([Some(PAWN_TO_E5), None], landing_attacked, 2 * bonus),
            ([None, Some(BISHOP_TO_C4)], landing_attacked, 2 * bonus),
            ([None, None], landing_attacked, bonus),
            (
                [Some(BISHOP_TO_C4), Some(PAWN_TO_E5)],
                landing_attacked,
                bonus,
            ),
            (full_context, Bitboard::empty(), 2 * bonus),
            (full_context, origin_attacked, 2 * bonus),
            (full_context, landing_attacked | origin_attacked, 2 * bonus),
            (
                full_context,
                landing_attacked | unrelated_attacked,
                3 * bonus,
            ),
            ([None, None], Bitboard::empty(), 0),
        ];

        for (probe_context, probe_threats, expected) in cases {
            let mut history = History::default();
            history.update_quiet(
                knight_move,
                KNIGHT_TO_F3,
                full_context,
                landing_attacked,
                bonus,
            );
            assert_eq!(
                history.quiet_score(knight_move, KNIGHT_TO_F3, probe_context, probe_threats),
                expected,
                "context {probe_context:?}, threats {probe_threats:?}"
            );
        }
    }

    #[test]
    fn steers_eval_correction_toward_search_score() {
        let tables = HASHED_CORRECTIONS as i32;
        let trained = CorrectionKeys {
            side: Color::White,
            hashed: std::array::from_fn(|table| {
                0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(table as u64 + 1)
            }),
        };
        let other_side = CorrectionKeys {
            side: Color::Black,
            ..trained
        };
        let mut other_first_key = trained;
        other_first_key.hashed[0] ^= u64::MAX;
        let partial = 100 * (tables - 1) / tables;
        let cases = [
            (
                1,
                15,
                100,
                trained,
                (1_600 * tables / 256)..=(1_600 * tables / 256),
            ),
            (
                1,
                0,
                100,
                trained,
                (100 * tables / 256)..=(100 * tables / 256),
            ),
            (200, 15, 100, trained, 97..=103),
            (200, 15, -30, trained, -33..=-27),
            (200, 15, 1_000, trained, (64 * tables)..=(64 * tables)),
            (200, 15, 100, other_side, 0..=0),
            (200, 15, 100, other_first_key, (partial - 6)..=(partial + 6)),
        ];

        for (repetitions, depth, search_score, probed, expected) in cases {
            let mut history = History::default();
            for _ in 0..repetitions {
                let residual = search_score - history.corrected_eval(&trained, 0);
                history.update_correction(&trained, depth, residual);
            }
            let correction = history.corrected_eval(&probed, 0);
            assert!(
                expected.contains(&correction),
                "x{repetitions}, depth {depth}, score {search_score}, probed {probed:?}: \
                 {correction}"
            );
        }
    }

    #[test]
    fn keeps_capture_history_bounded_and_separate() {
        let cases = [
            (1, Some(Piece::Queen), Some(Piece::Queen), history_bonus(3)),
            (1, Some(Piece::Queen), Some(Piece::Rook), 0),
            (1, None, None, history_bonus(3)),
            (1, Some(Piece::Pawn), None, 0),
            (500, Some(Piece::Rook), Some(Piece::Rook), MAX_HISTORY),
        ];

        for (repetitions, rewarded, probed, expected) in cases {
            let mut history = History::default();
            for _ in 0..repetitions {
                history.update_capture(KNIGHT_TO_F3, rewarded, history_bonus(3));
            }
            let score = history.capture_score(KNIGHT_TO_F3, probed);
            assert!(
                (expected - score).abs() <= MAX_HISTORY / 100,
                "rewarded {rewarded:?} x{repetitions}, probed {probed:?}: {score}"
            );
            history.clear();
            assert_eq!(history.capture_score(KNIGHT_TO_F3, probed), 0);
        }
    }
}
