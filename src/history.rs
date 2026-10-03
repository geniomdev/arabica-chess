use crate::types::{Color, Move, PIECE_TYPES, Piece, SIDES, SQUARES, Square};

pub const CONTINUATION_PLIES: usize = 2;

const MAX_HISTORY: i32 = 16_384;
const MAX_HISTORY_BONUS: i32 = 1_200;
const PIECE_TO_SLOTS: usize = SIDES * PIECE_TYPES * SQUARES;
const VICTIM_SLOTS: usize = PIECE_TYPES + 1;

pub type Continuations = [Option<PieceTo>; CONTINUATION_PLIES];

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
}

impl Default for History {
    fn default() -> Self {
        Self {
            butterfly: vec![0; SIDES * SQUARES * SQUARES],
            continuation: std::array::from_fn(|_| vec![0; PIECE_TO_SLOTS * PIECE_TO_SLOTS]),
            capture: vec![0; PIECE_TO_SLOTS * VICTIM_SLOTS],
        }
    }
}

pub fn history_bonus(depth: i32) -> i32 {
    (16 * depth * depth).min(MAX_HISTORY_BONUS)
}

fn apply_gravity(entry: &mut i32, bonus: i32) {
    *entry += bonus - *entry * bonus.abs() / MAX_HISTORY;
}

fn butterfly_index(side: Color, quiet: Move) -> usize {
    (side.index() * SQUARES + quiet.from().index()) * SQUARES + quiet.to().index()
}

fn continuation_index(previous: PieceTo, moved: PieceTo) -> usize {
    previous.index() * PIECE_TO_SLOTS + moved.index()
}

fn capture_index(moved: PieceTo, victim: Option<Piece>) -> usize {
    moved.index() * VICTIM_SLOTS + victim.map_or(PIECE_TYPES, Piece::index)
}

impl History {
    pub fn clear(&mut self) {
        self.butterfly.fill(0);
        for table in &mut self.continuation {
            table.fill(0);
        }
        self.capture.fill(0);
    }

    pub fn quiet_score(&self, quiet: Move, moved: PieceTo, continuations: &Continuations) -> i32 {
        let continuation_score: i32 = self
            .continuation
            .iter()
            .zip(continuations)
            .filter_map(|(table, previous)| {
                previous.map(|previous| table[continuation_index(previous, moved)])
            })
            .sum();
        self.butterfly[butterfly_index(moved.side, quiet)] + continuation_score
    }

    pub fn capture_score(&self, moved: PieceTo, victim: Option<Piece>) -> i32 {
        self.capture[capture_index(moved, victim)]
    }

    pub fn update_quiet(
        &mut self,
        quiet: Move,
        moved: PieceTo,
        continuations: &Continuations,
        bonus: i32,
    ) {
        apply_gravity(
            &mut self.butterfly[butterfly_index(moved.side, quiet)],
            bonus,
        );
        for (table, previous) in self.continuation.iter_mut().zip(continuations) {
            if let Some(previous) = *previous {
                apply_gravity(&mut table[continuation_index(previous, moved)], bonus);
            }
        }
    }

    pub fn update_capture(&mut self, moved: PieceTo, victim: Option<Piece>, bonus: i32) {
        apply_gravity(&mut self.capture[capture_index(moved, victim)], bonus);
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
        let bonus = history_bonus(4);
        let cases = [
            (full_context, 3 * bonus),
            ([Some(PAWN_TO_E5), None], 2 * bonus),
            ([None, Some(BISHOP_TO_C4)], 2 * bonus),
            ([None, None], bonus),
            ([Some(BISHOP_TO_C4), Some(PAWN_TO_E5)], bonus),
        ];

        for (probe_context, expected) in cases {
            let mut history = History::default();
            history.update_quiet(knight_move, KNIGHT_TO_F3, &full_context, bonus);
            assert_eq!(
                history.quiet_score(knight_move, KNIGHT_TO_F3, &probe_context),
                expected,
                "context {probe_context:?}"
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
