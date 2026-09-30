pub mod params;

use crate::attacks::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, queen_attacks, rook_attacks,
};
use crate::board::Board;
use crate::types::{Bitboard, Color, Piece, Square};

use params::*;

pub const PHASE_TOTAL: i32 = 24;
const PHASE_WEIGHTS: [i32; 6] = [0, 4, 2, 1, 1, 0];
const KING_ATTACKERS: [Piece; 4] = [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Score {
    pub mg: i32,
    pub eg: i32,
}

pub const fn s(mg: i32, eg: i32) -> Score {
    Score { mg, eg }
}

impl std::ops::AddAssign for Score {
    fn add_assign(&mut self, rhs: Self) {
        self.mg += rhs.mg;
        self.eg += rhs.eg;
    }
}

impl std::ops::Mul<i32> for Score {
    type Output = Self;

    fn mul(self, rhs: i32) -> Self {
        s(self.mg * rhs, self.eg * rhs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Term {
    Material(Piece),
    PieceSquare(Piece, usize),
    Mobility(Piece, usize),
    PassedPawn(usize),
    DoubledPawn,
    IsolatedPawn,
    BishopPair,
    RookOpenFile,
    RookSemiOpenFile,
    PawnShield,
    KingZoneAttacks(usize),
}

impl Term {
    pub fn weight(self) -> Score {
        match self {
            Self::Material(piece) => MATERIAL[piece.index()],
            Self::PieceSquare(piece, index) => PIECE_SQUARE[piece.index()][index],
            Self::Mobility(Piece::Knight, count) => KNIGHT_MOBILITY[count],
            Self::Mobility(Piece::Bishop, count) => BISHOP_MOBILITY[count],
            Self::Mobility(Piece::Rook, count) => ROOK_MOBILITY[count],
            Self::Mobility(_, count) => QUEEN_MOBILITY[count],
            Self::PassedPawn(rank) => PASSED_PAWN[rank],
            Self::DoubledPawn => DOUBLED_PAWN,
            Self::IsolatedPawn => ISOLATED_PAWN,
            Self::BishopPair => BISHOP_PAIR,
            Self::RookOpenFile => ROOK_OPEN_FILE,
            Self::RookSemiOpenFile => ROOK_SEMI_OPEN_FILE,
            Self::PawnShield => PAWN_SHIELD,
            Self::KingZoneAttacks(attacker) => KING_ZONE_ATTACKS[attacker],
        }
    }
}

pub trait Terms {
    fn add(&mut self, side: Color, term: Term, count: i32);
}

#[derive(Default)]
struct Totals(Score);

impl Terms for Totals {
    fn add(&mut self, side: Color, term: Term, count: i32) {
        let sign = match side {
            Color::White => count,
            Color::Black => -count,
        };
        self.0 += term.weight() * sign;
    }
}

pub fn phase(board: &Board) -> i32 {
    Piece::ALL
        .into_iter()
        .map(|piece| board.pieces(piece).count() as i32 * PHASE_WEIGHTS[piece.index()])
        .sum::<i32>()
        .min(PHASE_TOTAL)
}

pub fn taper(score: Score, phase: i32) -> i32 {
    (score.mg * phase + score.eg * (PHASE_TOTAL - phase)) / PHASE_TOTAL
}

pub fn evaluate(board: &Board) -> i32 {
    let mut totals = Totals::default();
    collect_terms(board, &mut totals);
    let white_score = taper(totals.0, phase(board));
    match board.state.active_color {
        Color::White => white_score,
        Color::Black => -white_score,
    }
}

pub fn collect_terms(board: &Board, terms: &mut impl Terms) {
    for side in Color::ALL {
        collect_side_terms(board, side, terms);
    }
}

fn relative_square(side: Color, square: Square) -> usize {
    match side {
        Color::White => square.index() ^ 56,
        Color::Black => square.index(),
    }
}

fn relative_rank(side: Color, square: Square) -> usize {
    match side {
        Color::White => square.rank() as usize,
        Color::Black => 7 - square.rank() as usize,
    }
}

fn ranks_ahead(side: Color, square: Square) -> Bitboard {
    let rank = u32::from(square.rank());
    match side {
        Color::White => Bitboard(u64::MAX.checked_shl(8 * (rank + 1)).unwrap_or(0)),
        Color::Black => Bitboard((1u64 << (8 * rank)) - 1),
    }
}

fn adjacent_files(file: u8) -> Bitboard {
    let west = if file > 0 {
        Bitboard::file(file - 1)
    } else {
        Bitboard::empty()
    };
    let east = if file < 7 {
        Bitboard::file(file + 1)
    } else {
        Bitboard::empty()
    };
    west | east
}

fn pawn_attack_span(board: &Board, side: Color) -> Bitboard {
    let mut attacked = Bitboard::empty();
    for square in board.pieces_of(side, Piece::Pawn) {
        attacked |= pawn_attacks(side, square);
    }
    attacked
}

fn collect_side_terms(board: &Board, side: Color, terms: &mut impl Terms) {
    let them = side.opponent();
    let occupancy = board.occupancy();
    let own_pawns = board.pieces_of(side, Piece::Pawn);
    let enemy_pawns = board.pieces_of(them, Piece::Pawn);
    let mobility_area = !(board.occupied_by(side) | pawn_attack_span(board, them));
    let enemy_king = board.king_square(them);
    let enemy_king_zone = king_attacks(enemy_king) | Bitboard::from_square(enemy_king);

    for piece in Piece::ALL {
        for square in board.pieces_of(side, piece) {
            terms.add(side, Term::Material(piece), 1);
            terms.add(
                side,
                Term::PieceSquare(piece, relative_square(side, square)),
                1,
            );
            let attacks = match piece {
                Piece::Knight => knight_attacks(square),
                Piece::Bishop => bishop_attacks(square, occupancy),
                Piece::Rook => rook_attacks(square, occupancy),
                Piece::Queen => queen_attacks(square, occupancy),
                Piece::King | Piece::Pawn => continue,
            };
            let mobility = (attacks & mobility_area).count() as usize;
            terms.add(side, Term::Mobility(piece, mobility), 1);
            let attacker = KING_ATTACKERS
                .iter()
                .position(|&candidate| candidate == piece)
                .expect("slider or knight attacks the king zone");
            let zone_hits = (attacks & enemy_king_zone).count() as i32;
            if zone_hits > 0 {
                terms.add(side, Term::KingZoneAttacks(attacker), zone_hits);
            }
        }
    }

    collect_pawn_terms(board, side, own_pawns, enemy_pawns, terms);

    if board.pieces_of(side, Piece::Bishop).count() >= 2 {
        terms.add(side, Term::BishopPair, 1);
    }
    for square in board.pieces_of(side, Piece::Rook) {
        let file = Bitboard::file(square.file());
        if (file & own_pawns).is_empty() {
            let term = if (file & enemy_pawns).is_empty() {
                Term::RookOpenFile
            } else {
                Term::RookSemiOpenFile
            };
            terms.add(side, term, 1);
        }
    }

    let king = board.king_square(side);
    let shield_files = Bitboard::file(king.file()) | adjacent_files(king.file());
    let shield_ranks = ranks_ahead(side, king) & !ranks_ahead(side, shifted_two_ranks(side, king));
    let shield = (own_pawns & shield_files & shield_ranks).count() as i32;
    if shield > 0 {
        terms.add(side, Term::PawnShield, shield);
    }
}

fn shifted_two_ranks(side: Color, king: Square) -> Square {
    let rank = match side {
        Color::White => (king.rank() + 2).min(7),
        Color::Black => king.rank().saturating_sub(2),
    };
    Square::new(king.file(), rank)
}

fn collect_pawn_terms(
    board: &Board,
    side: Color,
    own_pawns: Bitboard,
    enemy_pawns: Bitboard,
    terms: &mut impl Terms,
) {
    for file in 0..8 {
        let on_file = (own_pawns & Bitboard::file(file)).count() as i32;
        if on_file > 1 {
            terms.add(side, Term::DoubledPawn, on_file - 1);
        }
    }
    for square in board.pieces_of(side, Piece::Pawn) {
        let neighbours = adjacent_files(square.file());
        if (own_pawns & neighbours).is_empty() {
            terms.add(side, Term::IsolatedPawn, 1);
        }
        let front_span = ranks_ahead(side, square) & (Bitboard::file(square.file()) | neighbours);
        if (enemy_pawns & front_span).is_empty() {
            terms.add(side, Term::PassedPawn(relative_rank(side, square)), 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{KIWIPETE, START_POSITION};

    #[test]
    fn evaluates_positions() {
        let cases = [
            (START_POSITION, 0..=0),
            ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 800..=1100),
            ("4k3/8/8/8/8/8/8/3QK3 b - - 0 1", -1100..=-800),
            ("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1", -30..=30),
            ("4k3/8/8/8/8/8/P7/4K3 w - - 0 1", 50..=250),
        ];

        for (fen, range) in cases {
            let board: Board = fen
                .parse()
                .unwrap_or_else(|error| panic!("fen {fen:?} rejected: {error}"));
            let score = evaluate(&board);
            assert!(range.contains(&score), "fen {fen:?}, score {score}");
        }
    }

    #[test]
    fn mirrored_positions_evaluate_equally() {
        let cases = [
            (
                KIWIPETE,
                "r3k2r/pppbbppp/2n2q1P/1P2p3/3pn3/BN2PNP1/P1PPQPB1/R3K2R b KQkq - 0 1",
            ),
            (
                "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1",
                "4k3/8/8/4p3/8/8/8/4K3 b - - 0 1",
            ),
            (
                "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 3",
            ),
            (
                "6k1/5ppp/8/1P6/8/2P5/2P2PPP/3R2K1 w - - 0 1",
                "3r2k1/2p2ppp/2p5/8/1p6/8/5PPP/6K1 b - - 0 1",
            ),
        ];

        for (fen, mirrored) in cases {
            let [score, mirrored_score] = [fen, mirrored].map(|fen| {
                let board: Board = fen
                    .parse()
                    .unwrap_or_else(|error| panic!("fen {fen:?} rejected: {error}"));
                evaluate(&board)
            });
            assert_eq!(score, mirrored_score, "fen {fen:?} vs {mirrored:?}");
        }
    }

    #[test]
    fn collects_pawn_structure_terms() {
        struct Counts(Vec<(Color, Term, i32)>);

        impl Terms for Counts {
            fn add(&mut self, side: Color, term: Term, count: i32) {
                self.0.push((side, term, count));
            }
        }

        let cases = [
            (
                "4k3/8/8/8/8/2P5/2P5/4K3 w - - 0 1",
                Color::White,
                Term::DoubledPawn,
                1,
            ),
            (
                "4k3/8/8/8/8/2P5/2P5/4K3 w - - 0 1",
                Color::White,
                Term::IsolatedPawn,
                2,
            ),
            (
                "4k3/8/8/1P6/8/8/8/4K3 w - - 0 1",
                Color::White,
                Term::PassedPawn(4),
                1,
            ),
            (
                "4k3/p7/8/1P6/8/8/8/4K3 w - - 0 1",
                Color::White,
                Term::PassedPawn(4),
                0,
            ),
            (
                "4k3/8/8/8/8/8/5PPP/6K1 w - - 0 1",
                Color::White,
                Term::PawnShield,
                3,
            ),
            (
                "6k1/5ppp/8/8/8/8/8/4K3 w - - 0 1",
                Color::Black,
                Term::PawnShield,
                3,
            ),
            (
                "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
                Color::White,
                Term::RookOpenFile,
                1,
            ),
            (
                "4k3/p7/8/8/8/8/8/R3K3 w - - 0 1",
                Color::White,
                Term::RookSemiOpenFile,
                1,
            ),
            (
                "4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1",
                Color::White,
                Term::BishopPair,
                1,
            ),
        ];

        for (fen, side, term, expected) in cases {
            let board: Board = fen
                .parse()
                .unwrap_or_else(|error| panic!("fen {fen:?} rejected: {error}"));
            let mut counts = Counts(Vec::new());
            collect_terms(&board, &mut counts);
            let total: i32 = counts
                .0
                .iter()
                .filter(|&&(counted_side, counted_term, _)| {
                    counted_side == side && counted_term == term
                })
                .map(|&(_, _, count)| count)
                .sum();
            assert_eq!(total, expected, "fen {fen:?}, {side:?} {term:?}");
        }
    }
}
