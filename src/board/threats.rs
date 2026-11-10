use crate::attacks::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks_from, rook_attacks,
};
use crate::board::Board;
use crate::types::{Bitboard, Move, Piece};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Threats {
    pub by_pawns: Bitboard,
    pub by_minors: Bitboard,
    pub by_rooks: Bitboard,
    pub all: Bitboard,
}

impl Threats {
    pub fn against(self, piece: Piece) -> Bitboard {
        match piece {
            Piece::Knight | Piece::Bishop => self.by_pawns,
            Piece::Rook => self.by_minors,
            Piece::Queen => self.by_rooks,
            Piece::Pawn | Piece::King => Bitboard::empty(),
        }
    }

    pub fn escapes(self, piece: Piece, quiet: Move) -> bool {
        let danger = self.against(piece);
        danger.contains(quiet.from()) && !danger.contains(quiet.to())
    }

    pub fn walks_into(self, piece: Piece, quiet: Move) -> bool {
        let danger = self.against(piece);
        !danger.contains(quiet.from()) && danger.contains(quiet.to())
    }
}

impl Board {
    pub fn threats(&self) -> Threats {
        let enemy = self.state.active_color.opponent();
        let occupancy = self.occupancy();

        let by_pawns = pawn_attacks_from(enemy, self.pieces_of(enemy, Piece::Pawn));

        let mut by_minors = by_pawns;
        for square in self.pieces_of(enemy, Piece::Knight) {
            by_minors |= knight_attacks(square);
        }
        for square in self.pieces_of(enemy, Piece::Bishop) {
            by_minors |= bishop_attacks(square, occupancy);
        }

        let mut by_rooks = by_minors;
        for square in self.pieces_of(enemy, Piece::Rook) {
            by_rooks |= rook_attacks(square, occupancy);
        }

        let mut all = by_rooks | king_attacks(self.king_square(enemy));
        for square in self.pieces_of(enemy, Piece::Queen) {
            all |= bishop_attacks(square, occupancy) | rook_attacks(square, occupancy);
        }

        Threats {
            by_pawns,
            by_minors,
            by_rooks,
            all,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::board::testing::board;
    use crate::board::{KIWIPETE, START_POSITION};
    use crate::types::{Bitboard, Piece, Square};

    #[test]
    fn layers_opponent_attacks_by_attacker_value() {
        let cases = [
            START_POSITION,
            KIWIPETE,
            "4k3/8/8/8/8/8/8/R3K2R b KQ - 0 1",
            "r3k2r/8/8/3pP3/8/8/8/4K3 w kq d6 0 2",
            "8/8/4k3/8/2q5/8/1P6/K7 w - - 0 1",
            "1r2k3/2n5/8/3b4/8/8/8/4K3 w - - 0 1",
        ];
        let layers = [
            vec![Piece::Pawn],
            vec![Piece::Pawn, Piece::Knight, Piece::Bishop],
            vec![Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook],
            vec![
                Piece::Pawn,
                Piece::Knight,
                Piece::Bishop,
                Piece::Rook,
                Piece::Queen,
                Piece::King,
            ],
        ];

        for fen in cases {
            let position = board(fen);
            let enemy = position.state.active_color.opponent();
            let threats = position.threats();
            let computed = [
                threats.by_pawns,
                threats.by_minors,
                threats.by_rooks,
                threats.all,
            ];
            for (attackers, layer) in layers.iter().zip(computed) {
                let attacking = attackers.iter().fold(Bitboard::empty(), |pieces, &piece| {
                    pieces | position.pieces_of(enemy, piece)
                });
                let expected = (0..64u8)
                    .map(Square::from_index)
                    .filter(|&square| {
                        !(position.attackers_to(square, position.occupancy()) & attacking)
                            .is_empty()
                    })
                    .fold(Bitboard::empty(), |attacked, square| {
                        attacked | Bitboard::from_square(square)
                    });
                assert_eq!(layer, expected, "{fen}, attackers {attackers:?}");
            }
        }
    }
}
