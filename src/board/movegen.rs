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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Generation {
    All,
    Noisy,
    Quiet,
}

impl Generation {
    fn includes_noisy(self) -> bool {
        self != Self::Quiet
    }

    fn includes_quiet(self) -> bool {
        self != Self::Noisy
    }
}

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

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [Move] {
        &mut self.moves[..self.len]
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

fn piece_reach(piece: Piece, from: Square, occupancy: Bitboard) -> Bitboard {
    match piece {
        Piece::Knight => knight_attacks(from),
        Piece::Bishop => bishop_attacks(from, occupancy),
        Piece::Rook => rook_attacks(from, occupancy),
        Piece::Queen => bishop_attacks(from, occupancy) | rook_attacks(from, occupancy),
        Piece::King => king_attacks(from),
        Piece::Pawn => Bitboard::empty(),
    }
}

impl Board {
    pub fn generate_pseudo_legal(&self, moves: &mut MoveList) {
        self.generate(moves, Generation::All);
    }

    pub fn generate_noisy(&self, moves: &mut MoveList) {
        self.generate(moves, Generation::Noisy);
    }

    pub fn generate_quiets(&self, moves: &mut MoveList) {
        self.generate(moves, Generation::Quiet);
    }

    pub fn is_pseudo_legal(&self, candidate: Move) -> bool {
        if candidate == Move::NULL {
            return false;
        }
        let side = self.state.active_color;
        let Some((owner, piece)) = self.colored_piece_on(candidate.from()) else {
            return false;
        };
        if owner != side {
            return false;
        }
        let allies = self.occupied_by(side);
        let enemies = self.occupied_by(side.opponent());
        let occupancy = allies | enemies;
        let castles = matches!(
            candidate.kind(),
            MoveKind::KingCastle | MoveKind::QueenCastle
        );
        if piece == Piece::Pawn || castles {
            let mut moves = MoveList::new();
            if piece == Piece::Pawn {
                self.generate_pawn_moves(&mut moves, side, enemies, occupancy, Generation::All);
            } else if piece == Piece::King {
                self.generate_castling(&mut moves, side, occupancy);
            }
            return moves.as_slice().contains(&candidate);
        }
        let to = candidate.to();
        let expected_kind = if enemies.contains(to) {
            MoveKind::Capture
        } else if allies.contains(to) {
            return false;
        } else {
            MoveKind::Quiet
        };
        candidate.kind() == expected_kind
            && piece_reach(piece, candidate.from(), occupancy).contains(to)
    }

    pub fn parse_move(&self, notation: &str) -> Option<Move> {
        let mut moves = MoveList::new();
        self.generate_pseudo_legal(&mut moves);
        moves
            .as_slice()
            .iter()
            .copied()
            .find(|candidate| candidate.to_string() == notation)
    }

    fn generate(&self, moves: &mut MoveList, generation: Generation) {
        let side = self.state.active_color;
        let allies = self.occupied_by(side);
        let enemies = self.occupied_by(side.opponent());
        let occupancy = allies | enemies;
        let targets = match generation {
            Generation::All => !allies,
            Generation::Noisy => enemies,
            Generation::Quiet => !occupancy,
        };

        self.generate_pawn_moves(moves, side, enemies, occupancy, generation);
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
        if generation.includes_quiet() {
            self.generate_castling(moves, side, occupancy);
        }
    }

    fn generate_pawn_moves(
        &self,
        moves: &mut MoveList,
        side: Color,
        enemies: Bitboard,
        occupancy: Bitboard,
        generation: Generation,
    ) {
        let pawns = self.pieces_of(side, Piece::Pawn);
        let empty = !occupancy;
        let (step, double_push_rank, promotion_rank) = match side {
            Color::White => (8i8, Bitboard::rank(2), Bitboard::rank(7)),
            Color::Black => (-8i8, Bitboard::rank(5), Bitboard::rank(0)),
        };
        let origin =
            |to: Square, distance: i8| Square::from_index((to.raw() as i8 - step * distance) as u8);
        let single = forward(pawns, side) & empty;
        let double = forward(single & double_push_rank, side) & empty;

        if generation.includes_quiet() {
            for to in single & !promotion_rank {
                moves.push(Move::new(origin(to, 1), to, MoveKind::Quiet));
            }
            for to in double {
                moves.push(Move::new(origin(to, 2), to, MoveKind::DoublePush));
            }
        }
        if !generation.includes_noisy() {
            return;
        }
        for to in single & promotion_rank {
            push_promotions(moves, origin(to, 1), to, QUIET_PROMOTIONS);
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
        let square = |file: u8| Square::from_index(base + file);
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
    use crate::board::testing::{board, square};
    use crate::board::{KIWIPETE, START_POSITION};

    const BOTH_CASTLES: &str = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
    const WHITE_EN_PASSANT: &str = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3";
    const F1_ATTACKED: &str = "4k3/8/8/8/8/8/5r2/R3K2R w KQ - 0 1";
    const KING_IN_CHECK: &str = "4k3/8/8/8/8/8/4r3/R3K2R w KQ - 0 1";
    const CASTLING_BLOCKED: &str = "4k3/8/8/8/8/8/8/RN2K1NR w KQ - 0 1";
    const NO_CASTLING_RIGHTS: &str = "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1";

    fn generate(fen: &str) -> Vec<Move> {
        let board = board(fen);
        let mut moves = MoveList::new();
        board.generate_pseudo_legal(&mut moves);
        moves.as_slice().to_vec()
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

    const STAGED_FENS: [&str; 9] = [
        START_POSITION,
        KIWIPETE,
        WHITE_EN_PASSANT,
        BOTH_CASTLES,
        F1_ATTACKED,
        KING_IN_CHECK,
        "3r3k/4P3/8/8/8/8/8/K7 w - - 0 1",
        "k7/8/8/8/8/8/3p4/4R2K b - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    ];

    fn sorted(mut moves: Vec<Move>) -> Vec<Move> {
        moves.sort_by_key(|candidate| format!("{candidate:?}"));
        moves
    }

    #[test]
    fn noisy_and_quiet_stages_partition_pseudo_legal_moves() {
        for fen in STAGED_FENS {
            let board = board(fen);
            let mut noisy = MoveList::new();
            board.generate_noisy(&mut noisy);
            let mut quiets = MoveList::new();
            board.generate_quiets(&mut quiets);
            let all = generate(fen);
            let is_noisy = |candidate: &Move| candidate.is_capture() || candidate.is_promotion();
            assert!(noisy.as_slice().iter().all(is_noisy), "fen {fen:?}");
            assert!(!quiets.as_slice().iter().any(is_noisy), "fen {fen:?}");
            assert_eq!(
                sorted([noisy.as_slice(), quiets.as_slice()].concat()),
                sorted(all),
                "fen {fen:?}"
            );
        }
    }

    #[test]
    fn validates_moves_from_other_positions() {
        let foreign_moves: Vec<Move> = STAGED_FENS
            .iter()
            .flat_map(|fen| generate(fen))
            .chain([Move::NULL])
            .collect();

        for fen in STAGED_FENS {
            let board = board(fen);
            let generated = generate(fen);
            for &candidate in &foreign_moves {
                assert_eq!(
                    board.is_pseudo_legal(candidate),
                    generated.contains(&candidate),
                    "fen {fen:?}, move {candidate} {:?}",
                    candidate.kind()
                );
            }
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
