use crate::attacks::{bishop_attacks, knight_attacks, pawn_attacks, rook_attacks};
use crate::board::Board;
use crate::board::makemove::{castling_rook_path, en_passant_victim};
use crate::types::{Bitboard, Move, MoveKind, Piece};

impl Board {
    pub fn gives_check(&self, candidate: Move) -> bool {
        let us = self.state.active_color;
        let king = self.king_square(us.opponent());
        let (from, to) = (candidate.from(), candidate.to());
        let Some(moving) = self.piece_on(from) else {
            return false;
        };
        let mut occupancy =
            (self.occupancy() & !Bitboard::from_square(from)) | Bitboard::from_square(to);
        if candidate.kind() == MoveKind::EnPassant {
            occupancy &= !Bitboard::from_square(en_passant_victim(to));
        }
        if let Some((rook_from, rook_to)) = castling_rook_path(candidate) {
            occupancy =
                (occupancy & !Bitboard::from_square(rook_from)) | Bitboard::from_square(rook_to);
            return rook_attacks(rook_to, occupancy).contains(king);
        }
        let landing = match candidate.promotion().unwrap_or(moving) {
            Piece::Pawn => pawn_attacks(us, to),
            Piece::Knight => knight_attacks(to),
            Piece::Bishop => bishop_attacks(to, occupancy),
            Piece::Rook => rook_attacks(to, occupancy),
            Piece::Queen => bishop_attacks(to, occupancy) | rook_attacks(to, occupancy),
            Piece::King => Bitboard::empty(),
        };
        if landing.contains(king) {
            return true;
        }
        let stayed = !Bitboard::from_square(from);
        let diagonal = self.diagonal_sliders(us) & stayed;
        let orthogonal = self.orthogonal_sliders(us) & stayed;
        !((bishop_attacks(king, occupancy) & diagonal)
            | (rook_attacks(king, occupancy) & orthogonal))
            .is_empty()
    }
}

#[cfg(test)]
mod tests {
    use crate::board::testing::board;
    use crate::board::{KIWIPETE, MoveList, START_POSITION};

    #[test]
    fn predicts_checks_of_every_legal_move() {
        let cases = [
            (START_POSITION, 0),
            (KIWIPETE, 0),
            ("4k3/8/8/8/8/8/8/RR4K1 w - - 0 1", 3),
            ("7k/8/6K1/8/8/8/8/1Q6 w - - 0 1", 4),
            ("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1", 8),
            ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", 2),
            ("5k2/8/8/8/8/8/8/4K2R w K - 0 1", 3),
            ("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 3),
            ("k7/2P5/8/8/8/8/8/4K3 w - - 0 1", 2),
            ("8/8/8/k1pP3R/8/8/8/4K3 w - c6 0 1", 1),
            ("8/8/8/KPp4r/8/8/8/4k3 w - c6 0 1", 0),
            ("4k3/8/4n3/8/8/8/4R3/4K3 b - - 0 1", 0),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
                0,
            ),
        ];

        for (fen, expected_checks) in cases {
            let mut position = board(fen);
            let mut moves = MoveList::new();
            position.generate_pseudo_legal(&mut moves);
            let mut checks = 0;
            for &candidate in moves.as_slice() {
                let predicted = position.gives_check(candidate);
                if !position.make_move(candidate) {
                    continue;
                }
                let actual = position.in_check();
                position.unmake_move();
                assert_eq!(predicted, actual, "fen {fen:?}, move {candidate}");
                checks += usize::from(actual);
            }
            assert_eq!(checks, expected_checks, "fen {fen:?}");
        }
    }
}
