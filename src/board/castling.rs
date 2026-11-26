use crate::board::Board;
use crate::types::{Bitboard, Castle, CastlingRights, Move, MoveKind, SQUARES, Square};

impl Board {
    pub fn with_castling_rook(&self, stored: Move) -> Move {
        let side = self.state.active_color;
        let right = match stored.kind() {
            MoveKind::KingCastle => CastlingRights::king_side(side),
            MoveKind::QueenCastle => CastlingRights::queen_side(side),
            _ => return stored,
        };
        Move::new(
            stored.from(),
            self.castling_setup.rook(right),
            stored.kind(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CastlingPath {
    pub castle: Move,
    pub must_be_empty: Bitboard,
    pub king_path: Bitboard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CastlingSetup {
    paths: [CastlingPath; 4],
    lost_on_touch: [CastlingRights; SQUARES],
}

impl CastlingSetup {
    pub const fn new() -> Self {
        let unused = CastlingPath {
            castle: Move::NULL,
            must_be_empty: Bitboard::empty(),
            king_path: Bitboard::empty(),
        };
        Self {
            paths: [unused; 4],
            lost_on_touch: [CastlingRights::NONE; SQUARES],
        }
    }

    pub fn enable(&mut self, right: CastlingRights, king: Square, rook: Square) {
        let kind = if right.has(CastlingRights::WHITE_KING.union(CastlingRights::BLACK_KING)) {
            MoveKind::KingCastle
        } else {
            MoveKind::QueenCastle
        };
        let castle = Move::new(king, rook, kind);
        let squares = Castle::of(castle).expect("castling move");
        let movers = Bitboard::from_square(king) | Bitboard::from_square(rook);
        let king_path = rank_span(squares.king_from, squares.king_to);
        let swept = king_path | rank_span(squares.rook_from, squares.rook_to);
        self.paths[right.slot()] = CastlingPath {
            castle,
            must_be_empty: swept & !movers,
            king_path,
        };
        for square in [king, rook] {
            self.lost_on_touch[square.index()] = self.lost_on_touch[square.index()].union(right);
        }
    }

    pub const fn path(&self, right: CastlingRights) -> &CastlingPath {
        &self.paths[right.slot()]
    }

    pub const fn rook(&self, right: CastlingRights) -> Square {
        self.paths[right.slot()].castle.to()
    }

    pub const fn rights_after(
        &self,
        rights: CastlingRights,
        from: Square,
        to: Square,
    ) -> CastlingRights {
        rights.without(self.lost_on_touch[from.index()].union(self.lost_on_touch[to.index()]))
    }
}

fn rank_span(first: Square, second: Square) -> Bitboard {
    debug_assert_eq!(first.rank(), second.rank());
    let (low, high) = (first.raw().min(second.raw()), first.raw().max(second.raw()));
    Bitboard((u64::MAX << low) & (u64::MAX >> (63 - high)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_inclusive_rank_segments() {
        let cases = [
            (Square::E1, Square::G1, 0x70),
            (Square::G1, Square::E1, 0x70),
            (Square::B1, Square::B1, 0x02),
            (Square::A8, Square::H8, 0xFF00_0000_0000_0000),
        ];

        for (first, second, expected) in cases {
            assert_eq!(
                rank_span(first, second),
                Bitboard(expected),
                "{first}..{second}"
            );
        }
    }
}
