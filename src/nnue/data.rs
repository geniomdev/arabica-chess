use super::{feature_index, relative_square};
use crate::board::Board;
use crate::types::{Bitboard, Color, Piece, Square};

pub const RECORD_BYTES: usize = 32;

const PIECE_SLOTS: usize = 16;
const OPPONENT_FLAG: u8 = 8;
const PIECE_CODE_MASK: u8 = 7;
const NIBBLE_BITS: usize = 4;
const NIBBLE_MASK: u8 = 0b1111;

const BULLET_PIECE_ORDER: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Outcome {
    Loss = 0,
    Draw = 1,
    Win = 2,
}

impl Outcome {
    pub const fn flipped(self) -> Self {
        match self {
            Self::Loss => Self::Win,
            Self::Draw => Self::Draw,
            Self::Win => Self::Loss,
        }
    }

    pub const fn for_side(self, side: Color) -> Self {
        match side {
            Color::White => self,
            Color::Black => self.flipped(),
        }
    }

    const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Loss),
            1 => Some(Self::Draw),
            2 => Some(Self::Win),
            _ => None,
        }
    }
}

fn piece_code(piece: Piece) -> u8 {
    BULLET_PIECE_ORDER
        .iter()
        .position(|&candidate| candidate == piece)
        .expect("every piece has a code") as u8
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    occupancy: u64,
    pieces: [u8; PIECE_SLOTS],
    pub score: i16,
    pub outcome: Outcome,
}

impl Record {
    pub fn new(board: &Board, score: i16, outcome: Outcome) -> Self {
        let us = board.state.active_color;
        let mut codes = [None; 64];
        for square in board.occupancy() {
            let (side, piece) = board.colored_piece_on(square).expect("occupied square");
            let opponent = if side == us { 0 } else { OPPONENT_FLAG };
            codes[relative_square(us, square).index()] = Some(opponent | piece_code(piece));
        }

        let mut occupancy = 0;
        let mut pieces = [0; PIECE_SLOTS];
        for (slot, (square, code)) in codes
            .iter()
            .enumerate()
            .filter_map(|(square, code)| code.map(|code| (square, code)))
            .enumerate()
        {
            occupancy |= 1 << square;
            pieces[slot / 2] |= code << (NIBBLE_BITS * (slot & 1));
        }
        Self {
            occupancy,
            pieces,
            score,
            outcome,
        }
    }

    pub fn with_outcome(self, outcome: Outcome) -> Self {
        Self { outcome, ..self }
    }

    pub fn pieces(&self) -> impl Iterator<Item = (Color, Piece, Square)> + '_ {
        Bitboard(self.occupancy)
            .into_iter()
            .enumerate()
            .map(|(slot, square)| {
                let code = (self.pieces[slot / 2] >> (NIBBLE_BITS * (slot & 1))) & NIBBLE_MASK;
                let side = if code & OPPONENT_FLAG == 0 {
                    Color::White
                } else {
                    Color::Black
                };
                let piece = BULLET_PIECE_ORDER[usize::from(code & PIECE_CODE_MASK)];
                (side, piece, square)
            })
    }

    pub fn features(&self, perspective: Color) -> impl Iterator<Item = usize> + '_ {
        self.pieces()
            .map(move |(side, piece, square)| feature_index(perspective, side, piece, square))
    }

    fn king_square(&self, side: Color) -> Square {
        self.pieces()
            .find(|&(owner, piece, _)| owner == side && piece == Piece::King)
            .map_or(Square::A1, |(_, _, square)| square)
    }

    pub fn to_bytes(self) -> [u8; RECORD_BYTES] {
        let mut bytes = [0; RECORD_BYTES];
        bytes[0..8].copy_from_slice(&self.occupancy.to_le_bytes());
        bytes[8..24].copy_from_slice(&self.pieces);
        bytes[24..26].copy_from_slice(&self.score.to_le_bytes());
        bytes[26] = self.outcome as u8;
        bytes[27] = self.king_square(Color::White).raw();
        bytes[28] = relative_square(Color::Black, self.king_square(Color::Black)).raw();
        bytes
    }

    pub fn from_bytes(bytes: &[u8; RECORD_BYTES]) -> Option<Self> {
        let occupancy = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        let pieces: [u8; PIECE_SLOTS] = bytes[8..24].try_into().ok()?;
        let score = i16::from_le_bytes(bytes[24..26].try_into().ok()?);
        let outcome = Outcome::from_byte(bytes[26])?;
        let slots = occupancy.count_ones() as usize;
        if slots > PIECE_SLOTS * 2 {
            return None;
        }
        let valid_codes = (0..slots).all(|slot| {
            let code = (pieces[slot / 2] >> (NIBBLE_BITS * (slot & 1))) & NIBBLE_MASK;
            usize::from(code & PIECE_CODE_MASK) < BULLET_PIECE_ORDER.len()
        });
        valid_codes.then_some(Self {
            occupancy,
            pieces,
            score,
            outcome,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::testing::board;
    use crate::board::{KIWIPETE, START_POSITION};
    use crate::nnue::active_features;

    #[test]
    fn start_position_matches_bullet_layout() {
        let cases = [
            START_POSITION,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        ];
        let mut expected = [0u8; RECORD_BYTES];
        expected[0..8].copy_from_slice(&0xFFFF_0000_0000_FFFFu64.to_le_bytes());
        expected[8..12].copy_from_slice(&[0x13, 0x42, 0x25, 0x31]);
        expected[16..20].copy_from_slice(&[0x88; 4]);
        expected[20..24].copy_from_slice(&[0x9B, 0xCA, 0xAD, 0xB9]);
        expected[24..26].copy_from_slice(&25i16.to_le_bytes());
        expected[26] = Outcome::Draw as u8;
        expected[27] = Square::E1.raw();
        expected[28] = Square::E1.raw();

        for fen in cases {
            let bytes = Record::new(&board(fen), 25, Outcome::Draw).to_bytes();
            assert_eq!(bytes, expected, "fen {fen:?}");
        }
    }

    fn sorted(features: impl Iterator<Item = usize>) -> Vec<usize> {
        let mut features: Vec<usize> = features.collect();
        features.sort_unstable();
        features
    }

    #[test]
    fn records_round_trip_features_score_and_outcome() {
        let cases = [
            (START_POSITION, 0, Outcome::Draw),
            (KIWIPETE, -137, Outcome::Loss),
            ("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1", 250, Outcome::Win),
            ("6k1/5p1p/4N3/8/8/8/8/4K1Q1 w - - 0 1", 900, Outcome::Win),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
                i16::MIN,
                Outcome::Loss,
            ),
        ];

        for (fen, score, outcome) in cases {
            let position = board(fen);
            let us = position.state.active_color;
            let record = Record::new(&position, score, outcome);
            let decoded = Record::from_bytes(&record.to_bytes()).expect("valid record");
            assert_eq!(decoded, record, "fen {fen:?}");
            assert_eq!(
                (decoded.score, decoded.outcome),
                (score, outcome),
                "fen {fen:?}"
            );
            for (perspective, real_side) in [(Color::White, us), (Color::Black, us.opponent())] {
                assert_eq!(
                    sorted(decoded.features(perspective)),
                    sorted(active_features(&position, real_side)),
                    "fen {fen:?}, perspective {perspective:?}"
                );
            }
        }
    }

    #[test]
    fn rejects_malformed_records() {
        let valid = Record::new(&board(START_POSITION), 0, Outcome::Draw).to_bytes();
        let mut bad_outcome = valid;
        bad_outcome[26] = 3;
        let mut bad_piece = valid;
        bad_piece[8] = 0x07;
        let mut crowded = valid;
        crowded[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
        let cases = [
            (valid, true),
            (bad_outcome, false),
            (bad_piece, false),
            (crowded, false),
        ];

        for (bytes, accepted) in cases {
            assert_eq!(
                Record::from_bytes(&bytes).is_some(),
                accepted,
                "bytes {bytes:?}"
            );
        }
    }

    #[test]
    fn outcomes_convert_between_sides() {
        let cases = [
            (Outcome::Win, Color::White, Outcome::Win),
            (Outcome::Win, Color::Black, Outcome::Loss),
            (Outcome::Draw, Color::Black, Outcome::Draw),
            (Outcome::Loss, Color::Black, Outcome::Win),
        ];

        for (white, side, expected) in cases {
            assert_eq!(white.for_side(side), expected, "{white:?} for {side:?}");
        }
    }
}
