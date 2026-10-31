pub mod data;
pub mod games;
mod simd;
mod stack;

use std::fmt;
use std::fs;
use std::io;
use std::path::Path;
use std::ptr;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::board::Board;
use crate::search::MATE_BOUND;
use crate::types::{Bitboard, Color, PIECE_TYPES, Piece, SIDES, SQUARES, Square};
use simd::OutputKernel;
pub(crate) use stack::AccumulatorStack;

pub const INPUTS: usize = SIDES * PIECE_TYPES * SQUARES;
pub const HIDDEN: usize = 1024;
pub const OUTPUT_BUCKETS: usize = 8;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const SCALE: i32 = 400;

const VERTICAL_FLIP: u8 = 56;
const PIECES_PER_BUCKET: usize = 32 / OUTPUT_BUCKETS;

pub const fn relative_square(perspective: Color, square: Square) -> Square {
    match perspective {
        Color::White => square,
        Color::Black => Square::from_index(square.raw() ^ VERTICAL_FLIP),
    }
}

pub const fn output_bucket(piece_count: usize) -> usize {
    let bucket = piece_count.saturating_sub(2) / PIECES_PER_BUCKET;
    if bucket < OUTPUT_BUCKETS {
        bucket
    } else {
        OUTPUT_BUCKETS - 1
    }
}

#[repr(C, align(64))]
pub struct Network {
    pub feature_weights: [[i16; HIDDEN]; INPUTS],
    pub feature_bias: [i16; HIDDEN],
    pub output_weights: [[i16; SIDES * HIDDEN]; OUTPUT_BUCKETS],
    pub output_bias: [i16; OUTPUT_BUCKETS],
}

const VALUE_BYTES: usize = std::mem::size_of::<i16>();
const MAX_EVALUATION: i32 = MATE_BOUND - 1;

pub const NETWORK_VALUES: usize = INPUTS * HIDDEN + HIDDEN + OUTPUT_BUCKETS * (SIDES * HIDDEN + 1);
pub const NETWORK_BYTES: usize = NETWORK_VALUES * VALUE_BYTES;

const EMBEDDED_NETWORK: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/networks/net3.nnue"));
const _: () = assert!(EMBEDDED_NETWORK.len() == NETWORK_BYTES);

struct ActiveNetwork {
    network: Box<Network>,
    output_kernel: OutputKernel,
}

impl ActiveNetwork {
    fn new(network: Box<Network>) -> Self {
        let output_kernel = if simd::fits_madd(network.output_weights.as_flattened()) {
            *simd::FASTEST_KERNEL
        } else {
            OutputKernel::Portable
        };
        Self {
            network,
            output_kernel,
        }
    }
}

static EMBEDDED: LazyLock<ActiveNetwork> = LazyLock::new(|| {
    ActiveNetwork::new(
        Network::from_bytes(EMBEDDED_NETWORK).expect("embedded network matches the layout"),
    )
});

static LOADED: AtomicPtr<ActiveNetwork> = AtomicPtr::new(ptr::null_mut());

fn active() -> &'static ActiveNetwork {
    let loaded = LOADED.load(Ordering::Acquire);
    if loaded.is_null() {
        &EMBEDDED
    } else {
        unsafe { &*loaded }
    }
}

pub fn network() -> &'static Network {
    &active().network
}

#[derive(Debug)]
pub enum NetworkLoadError {
    Io(io::Error),
    Size(usize),
}

impl fmt::Display for NetworkLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Size(bytes) => write!(f, "expected {NETWORK_BYTES} bytes, got {bytes}"),
        }
    }
}

pub fn load_network(path: &Path) -> Result<(), NetworkLoadError> {
    let bytes = fs::read(path).map_err(NetworkLoadError::Io)?;
    let network = Network::from_bytes(&bytes).ok_or(NetworkLoadError::Size(bytes.len()))?;
    let loaded = Box::leak(Box::new(ActiveNetwork::new(network)));
    LOADED.store(loaded, Ordering::Release);
    Ok(())
}

pub fn use_embedded_network() {
    LOADED.store(ptr::null_mut(), Ordering::Release);
}

pub fn evaluate(board: &mut Board) -> i32 {
    evaluate_with(active(), board)
}

fn evaluate_with(active: &ActiveNetwork, board: &mut Board) -> i32 {
    if board.has_insufficient_material() {
        return 0;
    }
    let us = board.state.active_color;
    let piece_count = board.occupancy().count() as usize;
    let accumulators = board.accumulators();
    active
        .network
        .output(
            active.output_kernel,
            &accumulators[us.index()],
            &accumulators[us.opponent().index()],
            piece_count,
        )
        .clamp(-MAX_EVALUATION, MAX_EVALUATION)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct Accumulator(pub [i16; HIDDEN]);

impl Accumulator {
    pub fn add(&mut self, network: &Network, feature: usize) {
        for (value, weight) in self.0.iter_mut().zip(&network.feature_weights[feature]) {
            *value += weight;
        }
    }

    #[cfg(test)]
    pub fn remove(&mut self, network: &Network, feature: usize) {
        for (value, weight) in self.0.iter_mut().zip(&network.feature_weights[feature]) {
            *value -= weight;
        }
    }
}

impl Network {
    pub fn zeroed() -> Box<Self> {
        unsafe { Box::<Self>::new_zeroed().assume_init() }
    }

    pub fn values(&self) -> impl Iterator<Item = i16> + '_ {
        self.feature_weights
            .as_flattened()
            .iter()
            .chain(&self.feature_bias)
            .chain(self.output_weights.as_flattened())
            .chain(&self.output_bias)
            .copied()
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut i16> {
        self.feature_weights
            .as_flattened_mut()
            .iter_mut()
            .chain(&mut self.feature_bias)
            .chain(self.output_weights.as_flattened_mut())
            .chain(&mut self.output_bias)
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

    pub fn evaluate(&self, us: &Accumulator, them: &Accumulator, piece_count: usize) -> i32 {
        self.output(OutputKernel::Portable, us, them, piece_count)
    }

    fn output(
        &self,
        kernel: OutputKernel,
        us: &Accumulator,
        them: &Accumulator,
        piece_count: usize,
    ) -> i32 {
        let bucket = output_bucket(piece_count);
        let (our_weights, their_weights) = self.output_weights[bucket].split_at(HIDDEN);
        let sum = simd::screlu_dot(kernel, &us.0, our_weights)
            + simd::screlu_dot(kernel, &them.0, their_weights);
        let output = sum / i64::from(QA) + i64::from(self.output_bias[bucket]);
        (output * i64::from(SCALE) / i64::from(QA * QB)) as i32
    }
}

pub fn feature_index(perspective: Color, side: Color, piece: Piece, square: Square) -> usize {
    let relative_side = usize::from(side != perspective);
    (relative_side * PIECE_TYPES + piece.index()) * SQUARES
        + relative_square(perspective, square).index()
}

pub fn active_features(
    perspective: Color,
    pieces_of: impl Fn(Color, Piece) -> Bitboard + Copy,
) -> impl Iterator<Item = usize> {
    Color::ALL.into_iter().flat_map(move |side| {
        Piece::ALL.into_iter().flat_map(move |piece| {
            pieces_of(side, piece)
                .into_iter()
                .map(move |square| feature_index(perspective, side, piece, square))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::color_flipped_fen;
    use crate::board::testing::{board, features, scratch_path};
    use crate::board::{KIWIPETE, MoveList, START_POSITION};

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
                    features(&board(fen), perspective),
                    features(&board(&flipped), perspective.opponent()),
                    "fen {fen:?}, perspective {perspective:?}"
                );
            }
        }
    }

    #[test]
    fn network_rows_are_cache_aligned() {
        let network = Network::zeroed();
        assert_eq!(network.values().count(), NETWORK_VALUES);
        assert_eq!(network.to_bytes().len(), NETWORK_BYTES);
        let rows = [
            ("first feature", network.feature_weights[0].as_ptr()),
            ("last feature", network.feature_weights[INPUTS - 1].as_ptr()),
            ("feature bias", network.feature_bias.as_ptr()),
            ("first output bucket", network.output_weights[0].as_ptr()),
            (
                "last output bucket",
                network.output_weights[OUTPUT_BUCKETS - 1].as_ptr(),
            ),
        ];

        for (name, row) in rows {
            assert_eq!(row.addr() % 64, 0, "{name}");
        }
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
            (255, 64, 0, 0, 409_600),
            (255, 0, -64, 0, -409_600),
            (0, 64, 64, full, SCALE),
            (-100, 64, 64, 0, 0),
            (510, 1, 0, 0, 6_400),
            (255, 64, 64, -full, 819_200 - SCALE),
            (255, 127, 127, full, 1_626_000),
        ];

        for (bias, our_weight, their_weight, output_bias, expected) in cases {
            let mut network = Network::zeroed();
            network.feature_bias = [bias; HIDDEN];
            for row in &mut network.output_weights {
                row[..HIDDEN].fill(our_weight);
                row[HIDDEN..].fill(their_weight);
            }
            network.output_bias = [output_bias; OUTPUT_BUCKETS];
            let accumulator = network.refresh([]);
            assert_eq!(
                network.evaluate(&accumulator, &accumulator, 32),
                expected,
                "bias {bias}, weights {our_weight}/{their_weight}, output bias {output_bias}"
            );
        }
    }

    #[test]
    fn output_buckets_follow_piece_count() {
        let cases = [(2, 0), (5, 0), (6, 1), (17, 3), (30, 7), (31, 7), (32, 7)];

        for (piece_count, expected) in cases {
            assert_eq!(output_bucket(piece_count), expected, "{piece_count} pieces");
        }
    }

    #[test]
    fn evaluation_uses_the_bucket_of_the_piece_count() {
        let mut network = Network::zeroed();
        for (bucket, bias) in network.output_bias.iter_mut().enumerate() {
            *bias = (bucket as i32 * QA * QB / 8) as i16;
        }
        let accumulator = network.refresh([]);
        let cases = [(3, 0), (10, 2), (21, 4), (32, 7)];

        for (piece_count, bucket) in cases {
            assert_eq!(
                network.evaluate(&accumulator, &accumulator, piece_count),
                bucket * SCALE / 8,
                "{piece_count} pieces"
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
        let features = features(&position, Color::White);

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

    fn assert_fresh_accumulators(position: &mut Board, context: &str) {
        let fresh =
            Color::ALL.map(|perspective| network().refresh(features(position, perspective)));
        let fen = position.to_fen();
        assert_eq!(position.accumulators(), &fresh, "{context}, fen {fen:?}");
    }

    #[derive(Clone, Copy, Debug)]
    enum Probe {
        EveryNode,
        LeavesOnly,
        NullMoves,
        Invalidations,
    }

    fn walk(position: &mut Board, depth: u32, probe: Probe, context: &str) -> u64 {
        if matches!(probe, Probe::Invalidations) {
            position.invalidate_accumulators();
        }
        if depth == 0 || matches!(probe, Probe::EveryNode) {
            assert_fresh_accumulators(position, context);
        }
        if depth == 0 {
            return 1;
        }
        if matches!(probe, Probe::NullMoves) && !position.in_check() {
            position.make_null_move();
            assert_fresh_accumulators(position, context);
            position.unmake_null_move();
        }
        let mut moves = MoveList::new();
        position.generate_pseudo_legal(&mut moves);
        let mut nodes = 0;
        for &candidate in moves.as_slice() {
            let before = position.clone();
            if position.make_move(candidate) {
                nodes += walk(position, depth - 1, probe, context);
                position.unmake_move();
            }
            assert_eq!(*position, before, "{context}, move {candidate} not undone");
        }
        nodes
    }

    #[test]
    fn incremental_accumulators_match_refreshes() {
        let cases = [
            (START_POSITION, 3, 8_902),
            (KIWIPETE, 2, 2_039),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2_812),
            (
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                2,
                264,
            ),
            (
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                2,
                1_486,
            ),
        ];
        let probes = [
            Probe::EveryNode,
            Probe::LeavesOnly,
            Probe::NullMoves,
            Probe::Invalidations,
        ];

        for (fen, depth, expected_nodes) in cases {
            for probe in probes {
                let mut position = board(fen);
                let context = format!("{fen} with {probe:?}");
                let nodes = walk(&mut position, depth, probe, &context);
                assert_eq!(nodes, expected_nodes, "{context}");
            }
        }
    }

    #[test]
    fn loads_networks_from_files() {
        let directory = scratch_path("load");
        fs::create_dir_all(&directory).expect("directory created");
        let cases = [
            ("embedded.nnue", Some(NETWORK_BYTES), Ok(())),
            (
                "truncated.nnue",
                Some(NETWORK_BYTES - 1),
                Err(Some(NETWORK_BYTES - 1)),
            ),
            ("empty.nnue", Some(0), Err(Some(0))),
            ("missing.nnue", None, Err(None)),
        ];
        let position = START_POSITION;
        let expected_score = evaluate(&mut board(position));

        for (name, length, expected) in cases {
            let path = directory.join(name);
            if let Some(length) = length {
                fs::write(&path, &EMBEDDED_NETWORK[..length]).expect("network written");
            }
            let outcome = load_network(&path).map_err(|error| match error {
                NetworkLoadError::Size(bytes) => Some(bytes),
                NetworkLoadError::Io(_) => None,
            });
            assert_eq!(outcome, expected, "file {name}");
            assert_eq!(
                evaluate(&mut board(position)),
                expected_score,
                "file {name}"
            );
        }
        assert!(!ptr::eq(network(), &*EMBEDDED.network));
        use_embedded_network();
        assert!(ptr::eq(network(), &*EMBEDDED.network));
        fs::remove_dir_all(&directory).expect("directory removed");
    }

    #[test]
    fn embedded_network_evaluates_symmetrically() {
        let cases = [
            START_POSITION,
            KIWIPETE,
            "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1",
            "6k1/5p1p/4N3/8/8/8/8/4K1Q1 b - - 0 1",
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        ];

        for fen in cases {
            let score = evaluate(&mut board(fen));
            let mirrored = evaluate(&mut board(&color_flipped_fen(fen)));
            assert_eq!(score, mirrored, "fen {fen:?}");
            assert!(score.abs() < MAX_EVALUATION, "fen {fen:?}, score {score}");
        }
    }

    #[test]
    fn embedded_network_values_material() {
        let cases = [
            (
                "rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                50..=300,
            ),
            (
                "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                250..=800,
            ),
            (
                "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                800..=2500,
            ),
            (
                "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
                -2500..=-800,
            ),
            (START_POSITION, -60..=60),
            ("8/8/8/8/8/8/8/k6K w - - 0 1", 0..=0),
            ("8/8/8/8/8/8/8/k5NK b - - 0 1", 0..=0),
            ("8/8/3b4/8/8/8/8/k6K w - - 0 1", 0..=0),
        ];

        for (fen, expected) in cases {
            let score = evaluate(&mut board(fen));
            assert!(expected.contains(&score), "fen {fen:?}, score {score}");
        }
    }

    fn embedded_with_output_weights(
        weight: Option<i16>,
        output_kernel: OutputKernel,
    ) -> ActiveNetwork {
        let mut network = Network::from_bytes(EMBEDDED_NETWORK).expect("embedded network");
        if let Some(weight) = weight {
            network.output_weights.as_flattened_mut().fill(weight);
        }
        ActiveNetwork {
            network,
            output_kernel,
        }
    }

    fn played(fen: &str, moves: &[&str]) -> Board {
        let mut position = board(fen);
        for notation in moves {
            let played = position.parse_move(notation).expect("move parsed");
            assert!(position.make_move(played), "{notation} is legal in {fen:?}");
        }
        position
    }

    #[test]
    fn evaluation_matches_reference_on_fresh_accumulators() {
        let cases: [(&str, &[&str]); 6] = [
            (START_POSITION, &[]),
            (START_POSITION, &["e2e4"]),
            (START_POSITION, &["e2e4", "d7d5", "e4d5", "d8d5", "b1c3"]),
            (KIWIPETE, &[]),
            (KIWIPETE, &["e1g1", "h3g2", "f3f6"]),
            ("8/P6k/8/8/8/K7/6p1/8 b - - 0 1", &["g2g1q", "a7a8n"]),
        ];
        let kernels = simd::supported_kernels();

        for (fen, moves) in cases {
            for &kernel in &kernels {
                let active = embedded_with_output_weights(None, kernel);
                let mut position = played(fen, moves);
                let us = position.state.active_color;
                let fresh = Color::ALL
                    .map(|perspective| active.network.refresh(features(&position, perspective)));
                let expected = active.network.evaluate(
                    &fresh[us.index()],
                    &fresh[us.opponent().index()],
                    position.occupancy().count() as usize,
                );
                assert_eq!(
                    evaluate_with(&active, &mut position),
                    expected,
                    "fen {fen:?}, moves {moves:?}, kernel {kernel:?}"
                );
            }
        }
    }

    #[test]
    fn evaluation_stays_below_mate_bound() {
        let cases = [
            (START_POSITION, Some(i16::MAX), MAX_EVALUATION),
            (START_POSITION, Some(i16::MIN + 1), -MAX_EVALUATION),
            (KIWIPETE, Some(i16::MAX), MAX_EVALUATION),
            ("8/8/8/8/8/8/8/k5NK b - - 0 1", Some(i16::MAX), 0),
        ];

        for (fen, weight, expected) in cases {
            let active = embedded_with_output_weights(weight, OutputKernel::Portable);
            let score = evaluate_with(&active, &mut board(fen));
            assert_eq!(score, expected, "fen {fen:?}, output weights {weight:?}");
            assert!(score.abs() < MATE_BOUND, "fen {fen:?}, score {score}");
        }
    }
}
