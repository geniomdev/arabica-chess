pub mod params;
mod pawn_cache;

pub use pawn_cache::PawnCache;

use crate::attacks::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, queen_attacks, rook_attacks,
};
use crate::board::{Board, see_value};
use crate::types::{Bitboard, Color, Piece, Square};

use params::*;

pub const PHASE_TOTAL: i32 = 24;
pub const SCALE_NORMAL: i32 = 64;
const PHASE_WEIGHTS: [i32; 6] = [0, 4, 2, 1, 1, 0];
const SCALE_OPPOSITE_BISHOPS: i32 = 32;
const SCALE_PAWNLESS_ROOK_EDGE: i32 = 16;
const SCALE_PAWNLESS_MINOR_EDGE: i32 = 0;
const DARK_SQUARES: Bitboard = Bitboard(0xAA55_AA55_AA55_AA55);

const KING_ATTACKER_UNITS: [i32; 6] = [0, 5, 3, 2, 2, 0];
const SAFE_CHECK_UNITS: [i32; 6] = [0, 4, 5, 3, 4, 0];
const KING_DANGER_MIN_ATTACKERS: i32 = 2;
const OUTPOST_RANKS: std::ops::RangeInclusive<usize> = 3..=5;
const PASSED_KING_DISTANCE_FIRST_RANK: usize = 2;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Term {
    Material(Piece),
    PieceSquare(Piece, usize),
    Mobility(Piece, usize),
    PassedPawn(usize),
    SupportedPawn(usize),
    PhalanxPawn(usize),
    PassedPawnBlocked(usize),
    KingDanger(usize),
    ThreatByPawn(Piece),
    ThreatByMinor(Piece),
    ThreatByRook(Piece),
    DoubledPawn,
    IsolatedPawn,
    BackwardPawn,
    PassedOwnKingDistance,
    PassedEnemyKingDistance,
    BishopPair,
    RookOpenFile,
    RookSemiOpenFile,
    KnightOutpost,
    BishopOutpost,
    PawnShield,
    Hanging,
    Tempo,
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
            Self::SupportedPawn(rank) => SUPPORTED_PAWN[rank],
            Self::PhalanxPawn(rank) => PHALANX_PAWN[rank],
            Self::PassedPawnBlocked(rank) => PASSED_PAWN_BLOCKED[rank],
            Self::KingDanger(units) => KING_DANGER[units],
            Self::ThreatByPawn(victim) => THREAT_BY_PAWN[victim.index()],
            Self::ThreatByMinor(victim) => THREAT_BY_MINOR[victim.index()],
            Self::ThreatByRook(victim) => THREAT_BY_ROOK[victim.index()],
            Self::DoubledPawn => DOUBLED_PAWN,
            Self::IsolatedPawn => ISOLATED_PAWN,
            Self::BackwardPawn => BACKWARD_PAWN,
            Self::PassedOwnKingDistance => PASSED_OWN_KING_DISTANCE,
            Self::PassedEnemyKingDistance => PASSED_ENEMY_KING_DISTANCE,
            Self::BishopPair => BISHOP_PAIR,
            Self::RookOpenFile => ROOK_OPEN_FILE,
            Self::RookSemiOpenFile => ROOK_SEMI_OPEN_FILE,
            Self::KnightOutpost => KNIGHT_OUTPOST,
            Self::BishopOutpost => BISHOP_OUTPOST,
            Self::PawnShield => PAWN_SHIELD,
            Self::Hanging => HANGING,
            Self::Tempo => TEMPO,
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

struct AttackMaps {
    by_piece: [[Bitboard; 6]; 2],
    all: [Bitboard; 2],
}

impl AttackMaps {
    fn new(board: &Board) -> Self {
        let occupancy = board.occupancy();
        let mut by_piece = [[Bitboard::empty(); 6]; 2];
        let mut all = [Bitboard::empty(); 2];
        for side in Color::ALL {
            for piece in Piece::ALL {
                for square in board.pieces_of(side, piece) {
                    let attacks = piece_attacks(side, piece, square, occupancy);
                    by_piece[side.index()][piece.index()] |= attacks;
                    all[side.index()] |= attacks;
                }
            }
        }
        Self { by_piece, all }
    }

    fn of(&self, side: Color, piece: Piece) -> Bitboard {
        self.by_piece[side.index()][piece.index()]
    }

    fn any(&self, side: Color) -> Bitboard {
        self.all[side.index()]
    }
}

pub fn phase(board: &Board) -> i32 {
    Piece::ALL
        .into_iter()
        .map(|piece| board.pieces(piece).count() as i32 * PHASE_WEIGHTS[piece.index()])
        .sum::<i32>()
        .min(PHASE_TOTAL)
}

pub fn taper(score: Score, phase: i32, endgame_scale: i32) -> i32 {
    let scaled_eg = score.eg * endgame_scale / SCALE_NORMAL;
    (score.mg * phase + scaled_eg * (PHASE_TOTAL - phase)) / PHASE_TOTAL
}

pub fn evaluate(board: &Board, pawn_cache: &mut PawnCache) -> i32 {
    let pawn_structure = pawn_cache.probe(board);
    let mut totals = Totals(pawn_structure.score);
    collect_piece_terms(board, pawn_structure.passed, &mut totals);
    let white_score = taper(totals.0, phase(board), endgame_scale(board));
    match board.state.active_color {
        Color::White => white_score,
        Color::Black => -white_score,
    }
}

pub fn collect_terms(board: &Board, terms: &mut impl Terms) {
    let passed = Color::ALL.map(|side| collect_pawn_structure(board, side, terms));
    collect_piece_terms(board, passed, terms);
}

fn collect_piece_terms(board: &Board, passed: [Bitboard; 2], terms: &mut impl Terms) {
    let attacks = AttackMaps::new(board);
    for side in Color::ALL {
        collect_side_terms(board, side, &attacks, terms);
        collect_threat_terms(board, side, &attacks, terms);
        collect_passed_pawn_terms(board, side, passed[side.index()], terms);
    }
    terms.add(board.state.active_color, Term::Tempo, 1);
}

pub fn endgame_scale(board: &Board) -> i32 {
    let non_pawn_material = |side: Color| -> i32 {
        [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen]
            .into_iter()
            .map(|piece| board.pieces_of(side, piece).count() as i32 * see_value(piece))
            .sum()
    };
    let pawns = |side: Color| board.pieces_of(side, Piece::Pawn).count() as i32;
    let material = |side: Color| non_pawn_material(side) + pawns(side) * see_value(Piece::Pawn);
    let strong = if material(Color::White) >= material(Color::Black) {
        Color::White
    } else {
        Color::Black
    };
    let weak = strong.opponent();
    if pawns(strong) == 0
        && non_pawn_material(strong) - non_pawn_material(weak) <= see_value(Piece::Bishop)
    {
        return if non_pawn_material(strong) < see_value(Piece::Rook) {
            SCALE_PAWNLESS_MINOR_EDGE
        } else {
            SCALE_PAWNLESS_ROOK_EDGE
        };
    }
    if has_opposite_bishops_only(board) {
        return SCALE_OPPOSITE_BISHOPS;
    }
    SCALE_NORMAL
}

fn has_opposite_bishops_only(board: &Board) -> bool {
    let heavy_or_knights =
        board.pieces(Piece::Knight) | board.pieces(Piece::Rook) | board.pieces(Piece::Queen);
    if !heavy_or_knights.is_empty() {
        return false;
    }
    let [white_bishops, black_bishops] =
        Color::ALL.map(|side| board.pieces_of(side, Piece::Bishop));
    white_bishops.count() == 1
        && black_bishops.count() == 1
        && (white_bishops & DARK_SQUARES).is_empty() != (black_bishops & DARK_SQUARES).is_empty()
}

fn piece_attacks(side: Color, piece: Piece, square: Square, occupancy: Bitboard) -> Bitboard {
    match piece {
        Piece::Pawn => pawn_attacks(side, square),
        Piece::Knight => knight_attacks(square),
        Piece::Bishop => bishop_attacks(square, occupancy),
        Piece::Rook => rook_attacks(square, occupancy),
        Piece::Queen => queen_attacks(square, occupancy),
        Piece::King => king_attacks(square),
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

fn stop_square(side: Color, pawn: Square) -> Square {
    let rank = match side {
        Color::White => pawn.rank() + 1,
        Color::Black => pawn.rank() - 1,
    };
    Square::new(pawn.file(), rank)
}

fn distance(from: Square, to: Square) -> i32 {
    let files = i32::from(from.file()).abs_diff(i32::from(to.file()));
    let ranks = i32::from(from.rank()).abs_diff(i32::from(to.rank()));
    files.max(ranks) as i32
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

fn is_outpost(side: Color, square: Square, own_pawns: Bitboard, enemy_pawns: Bitboard) -> bool {
    let supported = !(pawn_attacks(side.opponent(), square) & own_pawns).is_empty();
    let challengers = enemy_pawns & adjacent_files(square.file()) & ranks_ahead(side, square);
    OUTPOST_RANKS.contains(&relative_rank(side, square)) && supported && challengers.is_empty()
}

fn collect_side_terms(board: &Board, side: Color, attacks: &AttackMaps, terms: &mut impl Terms) {
    let them = side.opponent();
    let occupancy = board.occupancy();
    let own_pawns = board.pieces_of(side, Piece::Pawn);
    let enemy_pawns = board.pieces_of(them, Piece::Pawn);
    let mobility_area = !(board.occupied_by(side) | attacks.of(them, Piece::Pawn));
    let enemy_king = board.king_square(them);
    let enemy_king_zone = king_attacks(enemy_king) | Bitboard::from_square(enemy_king);
    let mut king_attackers = 0;
    let mut king_danger_units = 0;

    for piece in Piece::ALL {
        for square in board.pieces_of(side, piece) {
            terms.add(side, Term::Material(piece), 1);
            terms.add(
                side,
                Term::PieceSquare(piece, relative_square(side, square)),
                1,
            );
            if matches!(piece, Piece::King | Piece::Pawn) {
                continue;
            }
            let piece_reach = piece_attacks(side, piece, square, occupancy);
            let mobility = (piece_reach & mobility_area).count() as usize;
            terms.add(side, Term::Mobility(piece, mobility), 1);
            let zone_hits = (piece_reach & enemy_king_zone).count() as i32;
            if zone_hits > 0 {
                king_attackers += 1;
                king_danger_units += KING_ATTACKER_UNITS[piece.index()] + zone_hits;
            }
            if is_outpost(side, square, own_pawns, enemy_pawns) {
                match piece {
                    Piece::Knight => terms.add(side, Term::KnightOutpost, 1),
                    Piece::Bishop => terms.add(side, Term::BishopOutpost, 1),
                    _ => {}
                }
            }
        }
    }

    let safe_squares = !(attacks.any(them) | board.occupied_by(side));
    let checking_squares = [
        (Piece::Knight, knight_attacks(enemy_king)),
        (Piece::Bishop, bishop_attacks(enemy_king, occupancy)),
        (Piece::Rook, rook_attacks(enemy_king, occupancy)),
        (Piece::Queen, queen_attacks(enemy_king, occupancy)),
    ];
    for (piece, squares) in checking_squares {
        if !(squares & attacks.of(side, piece) & safe_squares).is_empty() {
            king_danger_units += SAFE_CHECK_UNITS[piece.index()];
        }
    }
    if king_attackers >= KING_DANGER_MIN_ATTACKERS {
        let units = (king_danger_units as usize).min(KING_DANGER.len() - 1);
        terms.add(side, Term::KingDanger(units), 1);
    }

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

fn collect_threat_terms(board: &Board, side: Color, attacks: &AttackMaps, terms: &mut impl Terms) {
    let them = side.opponent();
    let enemy_pieces =
        board.occupied_by(them) & !board.pieces(Piece::Pawn) & !board.pieces(Piece::King);
    let undefended = !attacks.any(them);
    let minor_targets = enemy_pieces | (board.pieces_of(them, Piece::Pawn) & undefended);
    let minor_attacks = attacks.of(side, Piece::Knight) | attacks.of(side, Piece::Bishop);
    let threat_groups = [
        (
            attacks.of(side, Piece::Pawn) & enemy_pieces,
            Term::ThreatByPawn as fn(Piece) -> Term,
        ),
        (minor_attacks & minor_targets, Term::ThreatByMinor),
        (
            attacks.of(side, Piece::Rook) & minor_targets,
            Term::ThreatByRook,
        ),
    ];
    for (threatened, term) in threat_groups {
        for square in threatened {
            let victim = board
                .piece_on(square)
                .expect("threatened square holds a piece");
            terms.add(side, term(victim), 1);
        }
    }
    let hanging = (enemy_pieces & attacks.any(side) & undefended).count() as i32;
    if hanging > 0 {
        terms.add(side, Term::Hanging, hanging);
    }
}

fn collect_passed_pawn_terms(board: &Board, side: Color, passed: Bitboard, terms: &mut impl Terms) {
    let own_king = board.king_square(side);
    let enemy_king = board.king_square(side.opponent());
    for pawn in passed {
        let rank = relative_rank(side, pawn);
        let stop = stop_square(side, pawn);
        if board.piece_on(stop).is_some() {
            terms.add(side, Term::PassedPawnBlocked(rank), 1);
        }
        let proximity_weight = rank.saturating_sub(PASSED_KING_DISTANCE_FIRST_RANK) as i32;
        if proximity_weight > 0 {
            terms.add(
                side,
                Term::PassedOwnKingDistance,
                distance(own_king, stop) * proximity_weight,
            );
            terms.add(
                side,
                Term::PassedEnemyKingDistance,
                distance(enemy_king, stop) * proximity_weight,
            );
        }
    }
}

fn shifted_two_ranks(side: Color, king: Square) -> Square {
    let rank = match side {
        Color::White => (king.rank() + 2).min(7),
        Color::Black => king.rank().saturating_sub(2),
    };
    Square::new(king.file(), rank)
}

fn collect_pawn_structure(board: &Board, side: Color, terms: &mut impl Terms) -> Bitboard {
    let them = side.opponent();
    let own_pawns = board.pieces_of(side, Piece::Pawn);
    let enemy_pawns = board.pieces_of(them, Piece::Pawn);
    for file in 0..8 {
        let on_file = (own_pawns & Bitboard::file(file)).count() as i32;
        if on_file > 1 {
            terms.add(side, Term::DoubledPawn, on_file - 1);
        }
    }
    let mut passed = Bitboard::empty();
    for square in own_pawns {
        let rank = relative_rank(side, square);
        let neighbours = adjacent_files(square.file());
        let isolated = (own_pawns & neighbours).is_empty();
        if isolated {
            terms.add(side, Term::IsolatedPawn, 1);
        }
        let front_span = ranks_ahead(side, square) & (Bitboard::file(square.file()) | neighbours);
        if (enemy_pawns & front_span).is_empty() {
            terms.add(side, Term::PassedPawn(rank), 1);
            passed |= Bitboard::from_square(square);
        }
        if !(pawn_attacks(them, square) & own_pawns).is_empty() {
            terms.add(side, Term::SupportedPawn(rank), 1);
        }
        if !(own_pawns & neighbours & Bitboard::rank(square.rank())).is_empty() {
            terms.add(side, Term::PhalanxPawn(rank), 1);
        }
        let potential_supporters = own_pawns & neighbours & !ranks_ahead(side, square);
        let stop_attacked =
            !(pawn_attacks(side, stop_square(side, square)) & enemy_pawns).is_empty();
        if !isolated && potential_supporters.is_empty() && stop_attacked {
            terms.add(side, Term::BackwardPawn, 1);
        }
    }
    passed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::testing::board;
    use crate::board::{KIWIPETE, START_POSITION};

    fn engine_eval(fen: &str) -> i32 {
        evaluate(&board(fen), &mut PawnCache::default())
    }

    #[test]
    fn evaluates_positions() {
        let cases = [
            (START_POSITION, 0..=40),
            ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 800..=1300),
            ("4k3/8/8/8/8/8/8/3QK3 b - - 0 1", -1300..=-800),
            ("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1", -40..=40),
            ("4k3/8/8/8/8/8/P7/4K3 w - - 0 1", 50..=250),
        ];

        for (fen, range) in cases {
            let score = engine_eval(fen);
            assert!(range.contains(&score), "fen {fen:?}, score {score}");
        }
    }

    fn color_flipped_fen(fen: &str) -> String {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        let swap_case = |text: &str| -> String {
            text.chars()
                .map(|symbol| {
                    if symbol.is_ascii_uppercase() {
                        symbol.to_ascii_lowercase()
                    } else {
                        symbol.to_ascii_uppercase()
                    }
                })
                .collect()
        };
        let placement: Vec<String> = fields[0].split('/').rev().map(swap_case).collect();
        let side = if fields[1] == "w" { "b" } else { "w" };
        let mut castling: Vec<char> = swap_case(fields[2]).chars().collect();
        castling.sort_by_key(|&right| "KQkq-".find(right));
        let en_passant = match fields[3].as_bytes() {
            [file, b'3'] => format!("{}6", *file as char),
            [file, b'6'] => format!("{}3", *file as char),
            _ => fields[3].to_string(),
        };
        format!(
            "{} {side} {} {en_passant} {}",
            placement.join("/"),
            castling.into_iter().collect::<String>(),
            fields[4..].join(" ")
        )
    }

    #[test]
    fn mirrored_positions_evaluate_equally() {
        let cases = [
            KIWIPETE,
            "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1",
            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",
            "6k1/5ppp/8/1P6/8/2P5/2P2PPP/3R2K1 w - - 0 1",
            "r1bq1rk1/pp3ppp/2n1pn2/3pN3/1bPP4/2N1P3/PP3PPP/R1BQKB1R w KQ - 0 8",
            "6k1/5p1p/4N3/8/8/8/8/4K1Q1 w - - 0 1",
            "4kb2/p7/8/8/8/8/PP6/3BK3 b - - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        ];

        for fen in cases {
            let mirrored = color_flipped_fen(fen);
            assert_eq!(
                engine_eval(fen),
                engine_eval(&mirrored),
                "fen {fen:?} vs {mirrored:?}"
            );
        }
    }

    #[test]
    fn scales_drawish_endgames() {
        let cases = [
            ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", SCALE_NORMAL),
            ("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1", SCALE_PAWNLESS_MINOR_EDGE),
            (
                "4k3/3pp3/8/8/8/8/8/2N1K3 w - - 0 1",
                SCALE_PAWNLESS_MINOR_EDGE,
            ),
            ("4kn2/8/8/8/8/8/8/R3K3 w - - 0 1", SCALE_PAWNLESS_ROOK_EDGE),
            ("4kb2/8/8/8/8/8/8/R3K3 b - - 0 1", SCALE_PAWNLESS_ROOK_EDGE),
            ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", SCALE_NORMAL),
            ("4k3/8/8/8/8/8/8/1NB1K3 w - - 0 1", SCALE_NORMAL),
            ("4kb2/p7/8/8/8/8/PP6/3BK3 w - - 0 1", SCALE_OPPOSITE_BISHOPS),
            ("4kb2/p7/8/8/8/8/PP6/2B1K3 w - - 0 1", SCALE_NORMAL),
            ("4kb2/p7/8/8/8/8/PP6/R2BK3 w - - 0 1", SCALE_NORMAL),
        ];

        for (fen, expected) in cases {
            assert_eq!(endgame_scale(&board(fen)), expected, "fen {fen:?}");
        }
    }

    #[test]
    fn collects_terms() {
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
            (
                "4k3/8/8/8/3P4/2P5/8/4K3 w - - 0 1",
                Color::White,
                Term::SupportedPawn(3),
                1,
            ),
            (
                "4k3/8/8/8/2PP4/8/8/4K3 w - - 0 1",
                Color::White,
                Term::PhalanxPawn(3),
                2,
            ),
            (
                "4k3/8/2p5/8/1P6/P7/8/4K3 w - - 0 1",
                Color::White,
                Term::BackwardPawn,
                0,
            ),
            (
                "4k3/8/8/2p5/8/1P6/P7/4K3 w - - 0 1",
                Color::White,
                Term::BackwardPawn,
                0,
            ),
            (
                "4k3/8/8/8/p1P5/8/1P6/4K3 w - - 0 1",
                Color::White,
                Term::BackwardPawn,
                1,
            ),
            (
                "4k3/8/4p3/4N3/3P4/8/8/4K3 w - - 0 1",
                Color::White,
                Term::KnightOutpost,
                1,
            ),
            (
                "4k3/8/5p2/4N3/3P4/8/8/4K3 w - - 0 1",
                Color::White,
                Term::KnightOutpost,
                0,
            ),
            (
                "4k3/8/8/4N3/8/8/8/4K3 w - - 0 1",
                Color::White,
                Term::KnightOutpost,
                0,
            ),
            (
                "4k3/8/2n5/1P6/8/8/8/4K3 w - - 0 1",
                Color::White,
                Term::ThreatByPawn(Piece::Knight),
                1,
            ),
            (
                "4k3/8/2r5/8/4B3/8/8/4K3 w - - 0 1",
                Color::White,
                Term::ThreatByMinor(Piece::Rook),
                1,
            ),
            (
                "4k3/1p6/2p5/8/4B3/8/8/4K3 w - - 0 1",
                Color::White,
                Term::ThreatByMinor(Piece::Pawn),
                0,
            ),
            (
                "4k3/8/2p5/8/4B3/8/8/4K3 w - - 0 1",
                Color::White,
                Term::ThreatByMinor(Piece::Pawn),
                1,
            ),
            (
                "4k3/8/8/8/R2q4/8/8/4K3 w - - 0 1",
                Color::White,
                Term::ThreatByRook(Piece::Queen),
                1,
            ),
            (
                "4k3/8/8/8/R2n4/8/8/4K3 w - - 0 1",
                Color::White,
                Term::Hanging,
                1,
            ),
            (
                "4k3/4p3/R2n4/8/8/8/8/4K3 w - - 0 1",
                Color::White,
                Term::Hanging,
                0,
            ),
            (
                "4k3/8/8/8/8/1P6/8/4K3 w - - 0 1",
                Color::White,
                Term::Tempo,
                1,
            ),
            (
                "4k3/8/8/8/8/1P6/8/4K3 b - - 0 1",
                Color::White,
                Term::Tempo,
                0,
            ),
            (
                "4k3/8/8/8/8/1P6/8/4K3 b - - 0 1",
                Color::Black,
                Term::Tempo,
                1,
            ),
            (
                "4k3/8/1P6/8/8/8/8/4K3 w - - 0 1",
                Color::White,
                Term::PassedOwnKingDistance,
                3 * 6,
            ),
            (
                "4k3/8/1P6/8/8/8/8/4K3 w - - 0 1",
                Color::White,
                Term::PassedEnemyKingDistance,
                3 * 3,
            ),
            (
                "4k3/8/1P6/8/8/8/8/4K3 w - - 0 1",
                Color::White,
                Term::PassedPawnBlocked(5),
                0,
            ),
            (
                "4k3/1n6/1P6/8/8/8/8/4K3 w - - 0 1",
                Color::White,
                Term::PassedPawnBlocked(5),
                1,
            ),
        ];

        for (fen, side, term, expected) in cases {
            let board = board(fen);
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

    #[test]
    fn grows_king_danger_with_attackers_and_checks() {
        struct Danger(Option<usize>);

        impl Terms for Danger {
            fn add(&mut self, side: Color, term: Term, _count: i32) {
                if let (Color::White, Term::KingDanger(units)) = (side, term) {
                    self.0 = Some(units);
                }
            }
        }

        let cases = [
            ("6k1/5ppp/8/8/8/8/8/4K3 w - - 0 1", None),
            ("6k1/5ppp/8/8/8/5N2/8/4KQ2 w - - 0 1", None),
            ("6k1/5ppp/8/6N1/8/8/8/4KQ2 w - - 0 1", Some(2 + 2 + 5 + 1)),
            (
                "6k1/5p1p/4N3/8/8/8/8/4K1Q1 w - - 0 1",
                Some(2 + 2 + 5 + 2 + 4),
            ),
        ];

        for (fen, expected) in cases {
            let mut danger = Danger(None);
            collect_terms(&board(fen), &mut danger);
            assert_eq!(danger.0, expected, "fen {fen:?}");
        }
    }

    #[test]
    fn pawn_cache_matches_uncached_terms() {
        let fens = [
            START_POSITION,
            KIWIPETE,
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",
        ];
        let mut cache = PawnCache::default();

        for fen in fens.iter().chain(&fens) {
            let board = board(fen);
            let mut uncached = Totals::default();
            collect_terms(&board, &mut uncached);
            let white_score = taper(uncached.0, phase(&board), endgame_scale(&board));
            let expected = match board.state.active_color {
                Color::White => white_score,
                Color::Black => -white_score,
            };
            assert_eq!(evaluate(&board, &mut cache), expected, "fen {fen:?}");
        }
    }
}
