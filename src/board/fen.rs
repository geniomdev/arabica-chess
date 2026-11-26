use std::fmt;
use std::str::FromStr;

use super::Board;
use crate::types::{Bitboard, CastlingRights, Color, FILES, Piece, RANKS, Square};

pub const START_POSITION: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
pub const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

const BACK_RANKS: Bitboard = Bitboard(0xFF00_0000_0000_00FF);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FenError {
    FieldCount(usize),
    Placement(String),
    ActiveColor(String),
    Castling(String),
    EnPassant(String),
    HalfmoveClock(String),
    FullmoveNumber(String),
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldCount(count) => write!(f, "expected 4 to 6 fields, found {count}"),
            Self::Placement(text) => write!(f, "invalid piece placement: {text}"),
            Self::ActiveColor(text) => write!(f, "invalid active color: {text}"),
            Self::Castling(text) => write!(f, "invalid castling rights: {text}"),
            Self::EnPassant(text) => write!(f, "invalid en passant square: {text}"),
            Self::HalfmoveClock(text) => write!(f, "invalid halfmove clock: {text}"),
            Self::FullmoveNumber(text) => write!(f, "invalid fullmove number: {text}"),
        }
    }
}

impl std::error::Error for FenError {}

impl FromStr for Board {
    type Err = FenError;

    fn from_str(fen: &str) -> Result<Self, FenError> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if !(4..=6).contains(&fields.len()) {
            return Err(FenError::FieldCount(fields.len()));
        }

        let mut board = Board::new();
        board.parse_placement(fields[0])?;
        board.validate_placement(fields[0])?;
        board.state.active_color = parse_active_color(fields[1])?;
        board.parse_castling(fields[2])?;
        board.state.en_passant = parse_en_passant(fields[3])?;
        board.state.halfmove_clock = match fields.get(4) {
            Some(text) => text
                .parse()
                .map_err(|_| FenError::HalfmoveClock(text.to_string()))?,
            None => 0,
        };
        board.state.fullmove_number = match fields.get(5) {
            Some(text) => text
                .parse()
                .ok()
                .filter(|&number: &u16| number > 0)
                .ok_or_else(|| FenError::FullmoveNumber(text.to_string()))?,
            None => 1,
        };
        if !board.en_passant_matches_placement() {
            return Err(FenError::EnPassant(fields[3].to_string()));
        }
        let pusher = board.state.active_color.opponent();
        board.state.en_passant = board
            .state
            .en_passant
            .filter(|&target| board.en_passant_capturable(pusher, target));
        board.state.zobrist_key = board.compute_zobrist_key();
        Ok(board)
    }
}

impl Board {
    fn parse_placement(&mut self, placement: &str) -> Result<(), FenError> {
        let error = || FenError::Placement(placement.to_string());
        let rows: Vec<&str> = placement.split('/').collect();
        if rows.len() != RANKS as usize {
            return Err(error());
        }

        for (row, text) in rows.iter().enumerate() {
            let rank = RANKS - 1 - row as u8;
            let mut file = 0;
            for symbol in text.chars() {
                if let Some(empty) = symbol.to_digit(10).filter(|n| (1..=8).contains(n)) {
                    file += empty as u8;
                    if file > FILES {
                        return Err(error());
                    }
                } else {
                    let (side, piece) = Piece::from_symbol(symbol).ok_or_else(error)?;
                    if file >= FILES {
                        return Err(error());
                    }
                    self.put_piece(side, piece, Square::new(file, rank));
                    file += 1;
                }
            }
            if file != FILES {
                return Err(error());
            }
        }
        Ok(())
    }

    fn validate_placement(&self, placement: &str) -> Result<(), FenError> {
        let single_kings = Color::ALL
            .into_iter()
            .all(|side| self.pieces_of(side, Piece::King).count() == 1);
        if !single_kings || !(self.pieces(Piece::Pawn) & BACK_RANKS).is_empty() {
            return Err(FenError::Placement(placement.to_string()));
        }
        Ok(())
    }

    fn parse_castling(&mut self, text: &str) -> Result<(), FenError> {
        if text == "-" {
            return Ok(());
        }
        for symbol in text.chars() {
            let (right, king, rook) = self
                .castling_right(symbol)
                .ok_or_else(|| FenError::Castling(text.to_string()))?;
            if self.state.castling.has(right) && self.castling_setup.rook(right) != rook {
                return Err(FenError::Castling(text.to_string()));
            }
            self.state.castling.insert(right);
            self.castling_setup.enable(right, king, rook);
        }
        Ok(())
    }

    fn castling_right(&self, symbol: char) -> Option<(CastlingRights, Square, Square)> {
        let side = if symbol.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let king = self.king_square(side);
        if king.rank() != back_rank(side) {
            return None;
        }
        let rooks = self.pieces_of(side, Piece::Rook) & Bitboard::rank(king.rank());
        let rook_on = |file: u8| {
            Some(Square::new(file, king.rank())).filter(|&square| rooks.contains(square))
        };
        let rook = match symbol.to_ascii_lowercase() {
            'k' => (king.file() + 1..FILES).rev().find_map(rook_on),
            'q' => (0..king.file()).find_map(rook_on),
            file @ 'a'..='h' => rook_on(file as u8 - b'a'),
            _ => None,
        }?;
        let right = if rook.file() > king.file() {
            CastlingRights::king_side(side)
        } else {
            CastlingRights::queen_side(side)
        };
        Some((right, king, rook))
    }

    fn castling_field(&self) -> String {
        let symbols: String = CastlingRights::SINGLES
            .into_iter()
            .filter(|&right| self.state.castling.has(right))
            .map(|right| self.castling_symbol(right))
            .collect();
        if symbols.is_empty() {
            "-".to_string()
        } else {
            symbols
        }
    }

    fn castling_symbol(&self, right: CastlingRights) -> char {
        let side = if right.has(CastlingRights::WHITE_KING.union(CastlingRights::WHITE_QUEEN)) {
            Color::White
        } else {
            Color::Black
        };
        let king_side = right == CastlingRights::king_side(side);
        let rook = self.castling_setup.rook(right);
        let rooks = self.pieces_of(side, Piece::Rook) & Bitboard::rank(rook.rank());
        let outermost = rooks.into_iter().all(|other| {
            if king_side {
                other.file() <= rook.file()
            } else {
                other.file() >= rook.file()
            }
        });
        let symbol = match (outermost, king_side) {
            (true, true) => 'k',
            (true, false) => 'q',
            (false, _) => (b'a' + rook.file()) as char,
        };
        match side {
            Color::White => symbol.to_ascii_uppercase(),
            Color::Black => symbol,
        }
    }

    fn en_passant_matches_placement(&self) -> bool {
        let Some(square) = self.state.en_passant else {
            return true;
        };
        let (target_rank, pawn_rank, pawn_side) = match self.state.active_color {
            Color::White => (5, 4, Color::Black),
            Color::Black => (2, 3, Color::White),
        };
        square.rank() == target_rank
            && self.piece_on(square).is_none()
            && self
                .pieces_of(pawn_side, Piece::Pawn)
                .contains(Square::new(square.file(), pawn_rank))
    }

    pub fn fen_without(&self, removed: &[Square]) -> String {
        let mut stripped = self.clone();
        for &square in removed {
            if let Some((side, piece)) = stripped.colored_piece_on(square) {
                stripped.remove_piece(side, piece, square);
            }
        }
        for right in CastlingRights::SINGLES {
            if removed.contains(&self.castling_setup.rook(right)) {
                stripped.state.castling = stripped.state.castling.without(right);
            }
        }
        stripped.state.en_passant = None;
        stripped.state.halfmove_clock = 0;
        stripped.to_fen()
    }

    pub fn to_fen(&self) -> String {
        let mut placement = String::new();
        for rank in (0..RANKS).rev() {
            let mut empty = 0u8;
            for file in 0..FILES {
                match self.colored_piece_on(Square::new(file, rank)) {
                    None => empty += 1,
                    Some((side, piece)) => {
                        if empty > 0 {
                            placement.push((b'0' + empty) as char);
                            empty = 0;
                        }
                        placement.push(piece.symbol(side));
                    }
                }
            }
            if empty > 0 {
                placement.push((b'0' + empty) as char);
            }
            if rank > 0 {
                placement.push('/');
            }
        }

        let active_color = match self.state.active_color {
            Color::White => 'w',
            Color::Black => 'b',
        };
        let en_passant = self
            .state
            .en_passant
            .map_or_else(|| "-".to_string(), |square| square.to_string());

        format!(
            "{placement} {active_color} {} {en_passant} {} {}",
            self.castling_field(),
            self.state.halfmove_clock,
            self.state.fullmove_number
        )
    }
}

const fn back_rank(side: Color) -> u8 {
    match side {
        Color::White => 0,
        Color::Black => RANKS - 1,
    }
}

fn parse_active_color(text: &str) -> Result<Color, FenError> {
    match text {
        "w" => Ok(Color::White),
        "b" => Ok(Color::Black),
        _ => Err(FenError::ActiveColor(text.to_string())),
    }
}

fn parse_en_passant(text: &str) -> Result<Option<Square>, FenError> {
    if text == "-" {
        return Ok(None);
    }
    text.parse()
        .map(Some)
        .map_err(|_| FenError::EnPassant(text.to_string()))
}

pub fn color_flipped_fen(fen: &str) -> String {
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
    castling.sort_by_key(char::is_ascii_lowercase);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::KIWIPETE;
    use crate::board::testing::board;

    #[test]
    fn color_flip_mirrors_every_field() {
        let cases = [
            (START_POSITION, START_POSITION.replace(" w ", " b ")),
            (
                KIWIPETE,
                "r3k2r/pppbbppp/2n2q1P/1P2p3/3pn3/BN2PNP1/P1PPQPB1/R3K2R b KQkq - 0 1".to_string(),
            ),
            (
                "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w Kq f6 0 3",
                "rnbqkbnr/pppp1ppp/8/8/3PpP2/8/PPP1P1PP/RNBQKBNR b Qk f3 0 3".to_string(),
            ),
            (
                "4k3/8/8/8/8/8/4P3/4K3 b - - 7 40",
                "4k3/4p3/8/8/8/8/8/4K3 w - - 7 40".to_string(),
            ),
            (
                "r3k3/8/8/8/8/8/8/R1R1K3 w Cq - 0 1",
                "r1r1k3/8/8/8/8/8/8/R3K3 b Qc - 0 1".to_string(),
            ),
        ];

        for (fen, expected) in cases {
            let flipped = color_flipped_fen(fen);
            assert_eq!(flipped, expected, "fen {fen:?}");
            assert_eq!(color_flipped_fen(&flipped), fen, "fen {fen:?}");
        }
    }

    const EN_PASSANT: &str = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";

    #[test]
    fn parses_valid_positions() {
        let cases = [
            (START_POSITION, START_POSITION),
            (KIWIPETE, KIWIPETE),
            (EN_PASSANT, EN_PASSANT),
            (
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            ),
            (
                "4k3/8/8/8/8/8/8/4K3 b - - 12 40",
                "4k3/8/8/8/8/8/8/4K3 b - - 12 40",
            ),
            ("4k3/8/8/8/8/8/8/4K3 w - -", "4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
            (
                "  4k3/8/8/8/8/8/8/4K3   w - - 0 1 ",
                "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            ),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w HAha - 0 1",
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            ),
            (
                "1r4kr/8/8/8/8/8/8/1R4KR w BHbh - 0 1",
                "1r4kr/8/8/8/8/8/8/1R4KR w KQkq - 0 1",
            ),
            (
                "r3k3/8/8/8/8/8/8/R1R1K3 w qC - 0 1",
                "r3k3/8/8/8/8/8/8/R1R1K3 w Cq - 0 1",
            ),
            (
                "4k3/8/8/8/8/8/8/R1R1K3 w C - 0 1",
                "4k3/8/8/8/8/8/8/R1R1K3 w C - 0 1",
            ),
            (
                "4k3/8/8/8/8/8/8/R1K1R1R1 w QE - 0 1",
                "4k3/8/8/8/8/8/8/R1K1R1R1 w EQ - 0 1",
            ),
        ];

        for (fen, expected) in cases {
            let board = board(fen);
            assert_eq!(board.to_fen(), expected, "fen {fen:?}");
        }
    }

    #[test]
    fn rejects_invalid_positions() {
        let lone_kings = "4k3/8/8/8/8/8/8/4K3";
        let cases = [
            ("8/8 w", FenError::FieldCount(2)),
            (
                "4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra",
                FenError::FieldCount(7),
            ),
            (
                "88888888/8/8/8/8/8/8/8 w - - 0 1",
                FenError::Placement("88888888/8/8/8/8/8/8/8".to_string()),
            ),
            (
                "4k3/8/8/8/8/8/8 w - - 0 1",
                FenError::Placement("4k3/8/8/8/8/8/8".to_string()),
            ),
            (
                "8/8/8/8/8/8/8/4K3 w - - 0 1",
                FenError::Placement("8/8/8/8/8/8/8/4K3".to_string()),
            ),
            (
                "P3k3/8/8/8/8/8/8/4K3 w - - 0 1",
                FenError::Placement("P3k3/8/8/8/8/8/8/4K3".to_string()),
            ),
            (
                &format!("{lone_kings} x - - 0 1"),
                FenError::ActiveColor("x".to_string()),
            ),
            (
                &format!("{lone_kings} w X - 0 1"),
                FenError::Castling("X".to_string()),
            ),
            (
                &format!("{lone_kings} w K - 0 1"),
                FenError::Castling("K".to_string()),
            ),
            (
                &format!("{lone_kings} w A - 0 1"),
                FenError::Castling("A".to_string()),
            ),
            (
                "4k3/8/8/8/8/8/4K3/R7 w Q - 0 1",
                FenError::Castling("Q".to_string()),
            ),
            (
                "4k3/8/8/8/8/8/8/4K1RR w GK - 0 1",
                FenError::Castling("GK".to_string()),
            ),
            (
                &format!("{lone_kings} w - z9 0 1"),
                FenError::EnPassant("z9".to_string()),
            ),
            (
                &format!("{lone_kings} w - e3 0 1"),
                FenError::EnPassant("e3".to_string()),
            ),
            (
                &format!("{lone_kings} w - e6 0 1"),
                FenError::EnPassant("e6".to_string()),
            ),
            (
                &format!("{lone_kings} w - - 256 1"),
                FenError::HalfmoveClock("256".to_string()),
            ),
            (
                &format!("{lone_kings} w - - 0 0"),
                FenError::FullmoveNumber("0".to_string()),
            ),
        ];

        for (fen, expected) in cases {
            assert_eq!(fen.parse::<Board>().err(), Some(expected), "fen {fen:?}");
        }
    }
}
