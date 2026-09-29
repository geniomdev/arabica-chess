use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};
use crate::board::Board;
use crate::types::{Bitboard, CastlingRights, Color, Move, MoveKind, Piece, Square};

pub const MAX_MOVES: usize = 256;

const QUIET_PROMOTIONS: [MoveKind; 4] = [
    MoveKind::QueenPromotion,
    MoveKind::KnightPromotion,
    MoveKind::RookPromotion,
    MoveKind::BishopPromotion,
];
const CAPTURE_PROMOTIONS: [MoveKind; 4] = [
    MoveKind::QueenPromotionCapture,
    MoveKind::KnightPromotionCapture,
    MoveKind::RookPromotionCapture,
    MoveKind::BishopPromotionCapture,
];

pub struct MoveList {
    moves: [Move; MAX_MOVES],
    len: usize,
}

impl MoveList {
    pub fn new() -> Self {
        Self {
            moves: [Move::NULL; MAX_MOVES],
            len: 0,
        }
    }

    fn push(&mut self, candidate: Move) {
        debug_assert!(self.len < MAX_MOVES);
        self.moves[self.len] = candidate;
        self.len += 1;
    }

    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

fn forward(pawns: Bitboard, side: Color) -> Bitboard {
    match side {
        Color::White => Bitboard(pawns.0 << 8),
        Color::Black => Bitboard(pawns.0 >> 8),
    }
}

fn push_targets(moves: &mut MoveList, from: Square, targets: Bitboard, enemies: Bitboard) {
    for to in targets & enemies {
        moves.push(Move::new(from, to, MoveKind::Capture));
    }
    for to in targets & !enemies {
        moves.push(Move::new(from, to, MoveKind::Quiet));
    }
}

fn push_promotions(moves: &mut MoveList, from: Square, to: Square, kinds: [MoveKind; 4]) {
    for kind in kinds {
        moves.push(Move::new(from, to, kind));
    }
}

impl Board {
    pub fn generate_pseudo_legal(&self, moves: &mut MoveList) {
        let side = self.state.active_color;
        let allies = self.occupied_by(side);
        let enemies = self.occupied_by(side.opponent());
        let occupancy = allies | enemies;
        let targets = !allies;

        self.generate_pawn_moves(moves, side, enemies, occupancy);
        for from in self.pieces_of(side, Piece::Knight) {
            push_targets(moves, from, knight_attacks(from) & targets, enemies);
        }
        for from in self.diagonal_sliders(side) {
            push_targets(
                moves,
                from,
                bishop_attacks(from, occupancy) & targets,
                enemies,
            );
        }
        for from in self.orthogonal_sliders(side) {
            push_targets(
                moves,
                from,
                rook_attacks(from, occupancy) & targets,
                enemies,
            );
        }
        let king = self.king_square(side);
        push_targets(moves, king, king_attacks(king) & targets, enemies);
        self.generate_castling(moves, side, occupancy);
    }

    fn generate_pawn_moves(
        &self,
        moves: &mut MoveList,
        side: Color,
        enemies: Bitboard,
        occupancy: Bitboard,
    ) {
        let pawns = self.pieces_of(side, Piece::Pawn);
        let empty = !occupancy;
        let (step, double_push_rank, promotion_rank) = match side {
            Color::White => (8i8, Bitboard::rank(2), Bitboard::rank(7)),
            Color::Black => (-8i8, Bitboard::rank(5), Bitboard::rank(0)),
        };
        let origin = |to: Square, distance: i8| Square((to.0 as i8 - step * distance) as u8);
        let single = forward(pawns, side) & empty;
        let double = forward(single & double_push_rank, side) & empty;

        for to in single & !promotion_rank {
            moves.push(Move::new(origin(to, 1), to, MoveKind::Quiet));
        }
        for to in single & promotion_rank {
            push_promotions(moves, origin(to, 1), to, QUIET_PROMOTIONS);
        }
        for to in double {
            moves.push(Move::new(origin(to, 2), to, MoveKind::DoublePush));
        }
        for from in pawns {
            let captures = pawn_attacks(side, from) & enemies;
            for to in captures & !promotion_rank {
                moves.push(Move::new(from, to, MoveKind::Capture));
            }
            for to in captures & promotion_rank {
                push_promotions(moves, from, to, CAPTURE_PROMOTIONS);
            }
        }
        if let Some(target) = self.state.en_passant {
            for from in pawn_attacks(side.opponent(), target) & pawns {
                moves.push(Move::new(from, target, MoveKind::EnPassant));
            }
        }
    }

    fn generate_castling(&self, moves: &mut MoveList, side: Color, occupancy: Bitboard) {
        let (base, king_side, queen_side) = match side {
            Color::White => (0, CastlingRights::WHITE_KING, CastlingRights::WHITE_QUEEN),
            Color::Black => (56, CastlingRights::BLACK_KING, CastlingRights::BLACK_QUEEN),
        };
        let enemy = side.opponent();
        let square = |file: u8| Square(base + file);
        let is_free = |files: &[u8]| files.iter().all(|&file| !occupancy.contains(square(file)));
        let is_safe = |files: &[u8]| {
            files
                .iter()
                .all(|&file| !self.is_attacked(square(file), enemy, occupancy))
        };

        if self.state.castling.has(king_side) && is_free(&[5, 6]) && is_safe(&[4, 5, 6]) {
            moves.push(Move::new(square(4), square(6), MoveKind::KingCastle));
        }
        if self.state.castling.has(queen_side) && is_free(&[1, 2, 3]) && is_safe(&[2, 3, 4]) {
            moves.push(Move::new(square(4), square(2), MoveKind::QueenCastle));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{KIWIPETE, START_POSITION};

    const BOTH_CASTLES: &str = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
    const WHITE_EN_PASSANT: &str = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3";
    const F1_ATTACKED: &str = "4k3/8/8/8/8/8/5r2/R3K2R w KQ - 0 1";
    const KING_IN_CHECK: &str = "4k3/8/8/8/8/8/4r3/R3K2R w KQ - 0 1";
    const CASTLING_BLOCKED: &str = "4k3/8/8/8/8/8/8/RN2K1NR w KQ - 0 1";
    const NO_CASTLING_RIGHTS: &str = "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1";

    fn generate(fen: &str) -> Vec<Move> {
        let board: Board = fen
            .parse()
            .unwrap_or_else(|error| panic!("fen {fen:?} rejected: {error}"));
        let mut moves = MoveList::new();
        board.generate_pseudo_legal(&mut moves);
        moves.as_slice().to_vec()
    }

    fn square(name: &str) -> Square {
        name.parse().expect("valid square")
    }

    fn encoded(notation: &str, kind: MoveKind) -> Move {
        Move::new(square(&notation[..2]), square(&notation[2..4]), kind)
    }

    fn promotions(notation: &str, kinds: [MoveKind; 4]) -> Vec<Move> {
        kinds.map(|kind| encoded(notation, kind)).to_vec()
    }

    #[test]
    fn counts_moves() {
        let cases = [
            (START_POSITION, 20),
            (KIWIPETE, 48),
            (
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                20,
            ),
        ];

        for (fen, expected) in cases {
            assert_eq!(generate(fen).len(), expected, "fen {fen:?}");
        }
    }

    #[test]
    fn counts_moves_of_kind() {
        let cases = [
            (START_POSITION, MoveKind::DoublePush, 8),
            (KIWIPETE, MoveKind::KingCastle, 1),
            (KIWIPETE, MoveKind::QueenCastle, 1),
            (KIWIPETE, MoveKind::EnPassant, 0),
        ];

        for (fen, kind, expected) in cases {
            let count = generate(fen)
                .into_iter()
                .filter(|candidate| candidate.flags() == kind as u8)
                .count();
            assert_eq!(count, expected, "fen {fen:?}, kind {kind:?}");
        }
    }

    #[test]
    fn generates_moves() {
        let cases = [
            (WHITE_EN_PASSANT, "e5f6", MoveKind::EnPassant),
            (
                "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2",
                "e4d3",
                MoveKind::EnPassant,
            ),
            (BOTH_CASTLES, "e1g1", MoveKind::KingCastle),
            (BOTH_CASTLES, "e1c1", MoveKind::QueenCastle),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
                "e8g8",
                MoveKind::KingCastle,
            ),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
                "e8c8",
                MoveKind::QueenCastle,
            ),
            (F1_ATTACKED, "e1c1", MoveKind::QueenCastle),
            (
                "4k3/8/8/8/8/8/1r6/R3K2R w KQ - 0 1",
                "e1c1",
                MoveKind::QueenCastle,
            ),
        ];

        for (fen, notation, kind) in cases {
            let expected = encoded(notation, kind);
            assert!(
                generate(fen).contains(&expected),
                "fen {fen:?}, move {notation} {kind:?}"
            );
        }
    }

    #[test]
    fn omits_moves() {
        let cases = [
            (WHITE_EN_PASSANT, "e5d6", MoveKind::EnPassant),
            (F1_ATTACKED, "e1g1", MoveKind::KingCastle),
            (KING_IN_CHECK, "e1g1", MoveKind::KingCastle),
            (KING_IN_CHECK, "e1c1", MoveKind::QueenCastle),
            (CASTLING_BLOCKED, "e1g1", MoveKind::KingCastle),
            (CASTLING_BLOCKED, "e1c1", MoveKind::QueenCastle),
            (NO_CASTLING_RIGHTS, "e1g1", MoveKind::KingCastle),
            (NO_CASTLING_RIGHTS, "e1c1", MoveKind::QueenCastle),
        ];

        for (fen, notation, kind) in cases {
            let forbidden = encoded(notation, kind);
            assert!(
                !generate(fen).contains(&forbidden),
                "fen {fen:?}, move {notation} {kind:?}"
            );
        }
    }

    #[test]
    fn generates_exact_moves_from_square() {
        let cases = [
            (
                "3r3k/4P3/8/8/8/8/8/K7 w - - 0 1",
                "e7",
                [
                    promotions("e7e8", QUIET_PROMOTIONS),
                    promotions("e7d8", CAPTURE_PROMOTIONS),
                ]
                .concat(),
            ),
            (
                "k7/8/8/8/8/8/3p4/4R2K b - - 0 1",
                "d2",
                [
                    promotions("d2d1", QUIET_PROMOTIONS),
                    promotions("d2e1", CAPTURE_PROMOTIONS),
                ]
                .concat(),
            ),
            ("4k3/8/8/8/8/4n3/4P3/4K3 w - - 0 1", "e2", vec![]),
            (
                "4k3/8/8/8/4n3/8/4P3/4K3 w - - 0 1",
                "e2",
                vec![encoded("e2e3", MoveKind::Quiet)],
            ),
        ];

        for (fen, from, mut expected) in cases {
            let mut actual: Vec<Move> = generate(fen)
                .into_iter()
                .filter(|candidate| candidate.from() == square(from))
                .collect();
            let key = |candidate: &Move| format!("{candidate:?}");
            actual.sort_by_key(key);
            expected.sort_by_key(key);
            assert_eq!(actual, expected, "fen {fen:?}, from {from}");
        }
    }
}
