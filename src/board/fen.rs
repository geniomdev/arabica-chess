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
        board.state.active_color = parse_active_color(fields[1])?;
        board.state.castling = parse_castling(fields[2])?;
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
        board.validate(&fields)?;
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

    fn validate(&self, fields: &[&str]) -> Result<(), FenError> {
        let placement_error = || FenError::Placement(fields[0].to_string());
        for side in Color::ALL {
            if self.pieces_of(side, Piece::King).count() != 1 {
                return Err(placement_error());
            }
        }
        if !(self.pieces(Piece::Pawn) & BACK_RANKS).is_empty() {
            return Err(placement_error());
        }
        if !self.castling_matches_placement() {
            return Err(FenError::Castling(fields[2].to_string()));
        }
        if !self.en_passant_matches_placement() {
            return Err(FenError::EnPassant(fields[3].to_string()));
        }
        Ok(())
    }

    fn castling_matches_placement(&self) -> bool {
        let requirements = [
            (
                CastlingRights::WHITE_KING,
                Color::White,
                Square::new(4, 0),
                Square::new(7, 0),
            ),
            (
                CastlingRights::WHITE_QUEEN,
                Color::White,
                Square::new(4, 0),
                Square::new(0, 0),
            ),
            (
                CastlingRights::BLACK_KING,
                Color::Black,
                Square::new(4, 7),
                Square::new(7, 7),
            ),
            (
                CastlingRights::BLACK_QUEEN,
                Color::Black,
                Square::new(4, 7),
                Square::new(0, 7),
            ),
        ];
        requirements.into_iter().all(|(right, side, king, rook)| {
            !self.state.castling.has(right)
                || (self.pieces_of(side, Piece::King).contains(king)
                    && self.pieces_of(side, Piece::Rook).contains(rook))
        })
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
            self.state.castling, self.state.halfmove_clock, self.state.fullmove_number
        )
    }
}

fn parse_active_color(text: &str) -> Result<Color, FenError> {
    match text {
        "w" => Ok(Color::White),
        "b" => Ok(Color::Black),
        _ => Err(FenError::ActiveColor(text.to_string())),
    }
}

fn parse_castling(text: &str) -> Result<CastlingRights, FenError> {
    if text == "-" {
        return Ok(CastlingRights::NONE);
    }
    let mut rights = CastlingRights::NONE;
    for symbol in text.chars() {
        let right = match symbol {
            'K' => CastlingRights::WHITE_KING,
            'Q' => CastlingRights::WHITE_QUEEN,
            'k' => CastlingRights::BLACK_KING,
            'q' => CastlingRights::BLACK_QUEEN,
            _ => return Err(FenError::Castling(text.to_string())),
        };
        rights.insert(right);
    }
    Ok(rights)
}

fn parse_en_passant(text: &str) -> Result<Option<Square>, FenError> {
    if text == "-" {
        return Ok(None);
    }
    text.parse()
        .map(Some)
        .map_err(|_| FenError::EnPassant(text.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EN_PASSANT: &str = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";

    #[test]
    fn parses_valid_positions() {
        let cases = [
            (START_POSITION, START_POSITION),
            (KIWIPETE, KIWIPETE),
            (EN_PASSANT, EN_PASSANT),
            (
                "4k3/8/8/8/8/8/8/4K3 b - - 12 40",
                "4k3/8/8/8/8/8/8/4K3 b - - 12 40",
            ),
            ("4k3/8/8/8/8/8/8/4K3 w - -", "4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
            (
                "  4k3/8/8/8/8/8/8/4K3   w - - 0 1 ",
                "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            ),
        ];

        for (fen, expected) in cases {
            let board: Board = fen
                .parse()
                .unwrap_or_else(|error| panic!("fen {fen:?} rejected: {error}"));
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
