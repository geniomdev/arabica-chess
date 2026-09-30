use crate::attacks::{bishop_attacks, rook_attacks};
use crate::board::Board;
use crate::types::{Bitboard, Move, MoveKind, Piece, Square};

const SEE_VALUES: [i32; 6] = [20_000, 900, 500, 330, 320, 100];
const CHEAPEST_FIRST: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

pub const fn see_value(piece: Piece) -> i32 {
    SEE_VALUES[piece.index()]
}

impl Board {
    pub fn see(&self, played: Move, threshold: i32) -> bool {
        if matches!(played.kind(), MoveKind::KingCastle | MoveKind::QueenCastle) {
            return threshold <= 0;
        }
        let (from, to) = (played.from(), played.to());
        let captured_value = match played.kind() {
            MoveKind::EnPassant => see_value(Piece::Pawn),
            _ => self.piece_on(to).map_or(0, see_value),
        };
        let mut swap = captured_value - threshold;
        if swap < 0 {
            return false;
        }
        let moving = self.piece_on(from).expect("piece on origin square");
        swap = see_value(moving) - swap;
        if swap <= 0 {
            return true;
        }

        let mut occupancy =
            self.occupancy() & !Bitboard::from_square(from) & !Bitboard::from_square(to);
        if played.kind() == MoveKind::EnPassant {
            occupancy &= !Bitboard::from_square(Square(to.0 ^ 8));
        }
        let diagonal = self.pieces(Piece::Bishop) | self.pieces(Piece::Queen);
        let orthogonal = self.pieces(Piece::Rook) | self.pieces(Piece::Queen);
        let mut attackers = self.attackers_to(to, occupancy);
        let mut side = self.state.active_color;
        let mut winning = true;

        loop {
            side = side.opponent();
            attackers &= occupancy;
            let side_attackers = attackers & self.occupied_by(side);
            if side_attackers.is_empty() {
                break;
            }
            winning = !winning;
            let (piece, candidates) = CHEAPEST_FIRST
                .into_iter()
                .map(|piece| (piece, side_attackers & self.pieces(piece)))
                .find(|(_, candidates)| !candidates.is_empty())
                .expect("attacker of some piece type");
            if piece == Piece::King {
                let defended = !(attackers & self.occupied_by(side.opponent())).is_empty();
                return if defended { !winning } else { winning };
            }
            swap = see_value(piece) - swap;
            if swap < i32::from(winning) {
                break;
            }
            occupancy &= !Bitboard::from_square(candidates.lowest_square());
            if matches!(piece, Piece::Pawn | Piece::Bishop | Piece::Queen) {
                attackers |= bishop_attacks(to, occupancy) & diagonal;
            }
            if matches!(piece, Piece::Rook | Piece::Queen) {
                attackers |= rook_attacks(to, occupancy) & orthogonal;
            }
        }
        winning
    }
}

#[cfg(test)]
mod tests {
    use crate::board::{Board, MoveList};

    #[test]
    fn static_exchange_evaluation() {
        let cases = [
            ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 0, true),
            ("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 0, true),
            ("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1", "d1d5", 0, false),
            ("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1", "d1d5", -800, true),
            ("4k3/8/8/3r4/8/8/3R4/3RK3 w - - 0 1", "d2d5", 0, true),
            ("3rk3/8/8/3r4/8/8/3R4/4K3 w - - 0 1", "d2d5", 0, true),
            ("3rk3/8/8/3r4/8/8/3R4/4K3 w - - 0 1", "d2d5", 1, false),
            ("4k3/8/8/3n4/8/4B3/8/4K3 w - - 0 1", "e3d4", 0, true),
            ("4k3/8/4p3/3n4/8/8/8/3QK3 w - - 0 1", "d1d5", 0, false),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 0, true),
            ("4k3/8/2b5/8/8/8/6Q1/4K2B w - - 0 1", "g2c6", 0, true),
            ("4k3/1b6/2b5/8/8/8/6Q1/4K3 w - - 0 1", "g2c6", 0, false),
            ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1g1", 0, true),
        ];

        for (fen, notation, threshold, expected) in cases {
            let board: Board = fen
                .parse()
                .unwrap_or_else(|error| panic!("fen {fen:?} rejected: {error}"));
            let mut moves = MoveList::new();
            board.generate_pseudo_legal(&mut moves);
            let played = *moves
                .as_slice()
                .iter()
                .find(|candidate| candidate.to_string() == notation)
                .unwrap_or_else(|| panic!("fen {fen:?}, move {notation} not generated"));
            assert_eq!(
                board.see(played, threshold),
                expected,
                "fen {fen:?}, move {notation}, threshold {threshold}"
            );
        }
    }
}
