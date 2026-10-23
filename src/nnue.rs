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

const VALUE_BYTES: usize = std::mem::size_of::<i16>();

fn screlu(value: i16) -> i32 {
    let clipped = i32::from(value).clamp(0, QA);
    clipped * clipped
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Accumulator(pub [i16; HIDDEN]);

impl Accumulator {
    pub fn add(&mut self, network: &Network, feature: usize) {
        for (value, weight) in self.0.iter_mut().zip(&network.feature_weights[feature]) {
            *value += weight;
        }
    }

    pub fn remove(&mut self, network: &Network, feature: usize) {
        for (value, weight) in self.0.iter_mut().zip(&network.feature_weights[feature]) {
            *value -= weight;
        }
    }
}

impl Network {
    pub fn zeroed() -> Box<Self> {
        Box::new(Self {
            feature_weights: [[0; HIDDEN]; INPUTS],
            feature_bias: [0; HIDDEN],
            output_weights: [0; SIDES * HIDDEN],
            output_bias: 0,
        })
    }

    pub fn values(&self) -> impl Iterator<Item = i16> + '_ {
        self.feature_weights
            .as_flattened()
            .iter()
            .chain(&self.feature_bias)
            .chain(&self.output_weights)
            .chain(std::iter::once(&self.output_bias))
            .copied()
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut i16> {
        self.feature_weights
            .as_flattened_mut()
            .iter_mut()
            .chain(&mut self.feature_bias)
            .chain(&mut self.output_weights)
            .chain(std::iter::once(&mut self.output_bias))
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Box<Self>> {
        if bytes.len() != NETWORK_BYTES {
            return None;
        }
        let mut network = Self::zeroed();
        for (value, chunk) in network.values_mut().zip(bytes.as_chunks::<VALUE_BYTES>().0) {
            *value = i16::from_le_bytes(*chunk);
        }
        Some(network)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.values().flat_map(i16::to_le_bytes).collect()
    }

    pub fn refresh(&self, features: impl IntoIterator<Item = usize>) -> Accumulator {
        let mut accumulator = Accumulator(self.feature_bias);
        for feature in features {
            accumulator.add(self, feature);
        }
        accumulator
    }

    pub fn evaluate(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        let (our_weights, their_weights) = self.output_weights.split_at(HIDDEN);
        let sum: i32 =
            us.0.iter()
                .zip(our_weights)
                .chain(them.0.iter().zip(their_weights))
                .map(|(&value, &weight)| screlu(value) * i32::from(weight))
                .sum();
        let output = i64::from(sum / QA + i32::from(self.output_bias));
        (output * i64::from(SCALE) / i64::from(QA * QB)) as i32
    }
}

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
        assert_eq!(Network::zeroed().values().count(), parameters);
    }

    #[test]
    fn networks_round_trip_through_bytes() {
        let mut network = Network::zeroed();
        for (index, value) in network.values_mut().enumerate() {
            *value = (index as i64 * 7919 % 65_536 - 32_768) as i16;
        }
        let bytes = network.to_bytes();
        let cases = [
            (bytes.clone(), true),
            (bytes[..NETWORK_BYTES - 1].to_vec(), false),
            ([bytes.as_slice(), &[0]].concat(), false),
        ];

        for (input, accepted) in cases {
            let decoded = Network::from_bytes(&input);
            assert_eq!(decoded.is_some(), accepted, "{} bytes", input.len());
            if let Some(decoded) = decoded {
                assert!(decoded.values().eq(network.values()), "values differ");
            }
        }
    }

    #[test]
    fn evaluates_hand_built_networks() {
        let full = (QA * QB) as i16;
        let cases = [
            (255, 64, 0, 0, 51_200),
            (255, 0, -64, 0, -51_200),
            (0, 64, 64, full, SCALE),
            (-100, 64, 64, 0, 0),
            (510, 1, 0, 0, 800),
            (255, 64, 64, -full, 102_400 - SCALE),
            (255, 127, 127, full, 203_600),
        ];

        for (bias, our_weight, their_weight, output_bias, expected) in cases {
            let mut network = Network::zeroed();
            network.feature_bias = [bias; HIDDEN];
            network.output_weights[..HIDDEN].fill(our_weight);
            network.output_weights[HIDDEN..].fill(their_weight);
            network.output_bias = output_bias;
            let accumulator = network.refresh([]);
            assert_eq!(
                network.evaluate(&accumulator, &accumulator),
                expected,
                "bias {bias}, weights {our_weight}/{their_weight}, output bias {output_bias}"
            );
        }
    }

    #[test]
    fn accumulators_add_and_remove_features() {
        let mut network = Network::zeroed();
        for (feature, row) in network.feature_weights.iter_mut().enumerate() {
            row.fill((feature % 17) as i16 - 8);
        }
        network.feature_bias = [3; HIDDEN];
        let position = board(KIWIPETE);
        let features: Vec<usize> = active_features(&position, Color::White).collect();

        let mut incremental = network.refresh([]);
        for &feature in &features {
            incremental.add(&network, feature);
        }
        assert_eq!(incremental, network.refresh(features.iter().copied()));
        for &feature in &features {
            incremental.remove(&network, feature);
        }
        assert_eq!(incremental, network.refresh([]));
    }
}
