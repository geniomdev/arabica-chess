#![allow(dead_code)]

pub mod data;

use crate::board::Board;
use crate::types::{Color, PIECE_TYPES, Piece, SIDES, SQUARES, Square};

pub const INPUTS: usize = SIDES * PIECE_TYPES * SQUARES;
pub const HIDDEN: usize = 128;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const SCALE: i32 = 400;

const VERTICAL_FLIP: u8 = 56;

pub const fn relative_square(perspective: Color, square: Square) -> Square {
    match perspective {
        Color::White => square,
        Color::Black => Square::from_index(square.raw() ^ VERTICAL_FLIP),
    }
}

#[repr(C)]
pub struct Network {
    pub feature_weights: [[i16; HIDDEN]; INPUTS],
    pub feature_bias: [i16; HIDDEN],
    pub output_weights: [i16; SIDES * HIDDEN],
    pub output_bias: i16,
}

pub const NETWORK_BYTES: usize = std::mem::size_of::<Network>();

pub fn feature_index(perspective: Color, side: Color, piece: Piece, square: Square) -> usize {
    let relative_side = usize::from(side != perspective);
    (relative_side * PIECE_TYPES + piece.index()) * SQUARES
        + relative_square(perspective, square).index()
}

pub fn active_features(board: &Board, perspective: Color) -> impl Iterator<Item = usize> + '_ {
    Color::ALL.into_iter().flat_map(move |side| {
        Piece::ALL.into_iter().flat_map(move |piece| {
            board
                .pieces_of(side, piece)
                .into_iter()
                .map(move |square| feature_index(perspective, side, piece, square))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::color_flipped_fen;
    use crate::board::testing::board;
    use crate::board::{KIWIPETE, START_POSITION};

    #[test]
    fn feature_indices_cover_inputs_exactly_once() {
        for perspective in Color::ALL {
            let mut hits = [0u8; INPUTS];
            for side in Color::ALL {
                for piece in Piece::ALL {
                    for square in 0..SQUARES as u8 {
                        let index =
                            feature_index(perspective, side, piece, Square::from_index(square));
                        hits[index] += 1;
                    }
                }
            }
            assert!(
                hits.iter().all(|&count| count == 1),
                "perspective {perspective:?}"
            );
        }
    }

    #[test]
    fn feature_indices_follow_layout() {
        let cases = [
            (Color::White, Color::White, Piece::King, Square::A1, 0),
            (
                Color::White,
                Color::White,
                Piece::Pawn,
                Square::H8,
                5 * 64 + 63,
            ),
            (Color::White, Color::Black, Piece::King, Square::A1, 384),
            (
                Color::White,
                Color::Black,
                Piece::Queen,
                Square::D8,
                384 + 64 + 59,
            ),
            (Color::Black, Color::Black, Piece::King, Square::E8, 4),
            (
                Color::Black,
                Color::Black,
                Piece::Pawn,
                Square::A7,
                5 * 64 + 8,
            ),
            (
                Color::Black,
                Color::White,
                Piece::King,
                Square::E1,
                384 + 60,
            ),
            (
                Color::Black,
                Color::White,
                Piece::Knight,
                Square::G1,
                384 + 4 * 64 + 62,
            ),
        ];

        for (perspective, side, piece, square, expected) in cases {
            assert_eq!(
                feature_index(perspective, side, piece, square),
                expected,
                "perspective {perspective:?}, {side:?} {piece:?} on {square:?}"
            );
        }
    }

    fn sorted_features(fen: &str, perspective: Color) -> Vec<usize> {
        let mut features: Vec<usize> = active_features(&board(fen), perspective).collect();
        features.sort_unstable();
        features
    }

    #[test]
    fn color_flip_swaps_perspectives() {
        let cases = [
            START_POSITION,
            KIWIPETE,
            "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1",
            "6k1/5p1p/4N3/8/8/8/8/4K1Q1 w - - 0 1",
            "4kb2/p7/8/8/8/8/PP6/3BK3 b - - 0 1",
        ];

        for fen in cases {
            let flipped = color_flipped_fen(fen);
            for perspective in Color::ALL {
                assert_eq!(
                    sorted_features(fen, perspective),
                    sorted_features(&flipped, perspective.opponent()),
                    "fen {fen:?}, perspective {perspective:?}"
                );
            }
        }
    }

    #[test]
    fn network_layout_has_no_padding() {
        let parameters = INPUTS * HIDDEN + HIDDEN + SIDES * HIDDEN + 1;
        assert_eq!(NETWORK_BYTES, parameters * std::mem::size_of::<i16>());
    }
}
