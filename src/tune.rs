use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::thread;

use crate::board::Board;
use crate::eval::params::{
    BISHOP_MOBILITY, KING_DANGER, KNIGHT_MOBILITY, PASSED_PAWN, PASSED_PAWN_BLOCKED, PHALANX_PAWN,
    QUEEN_MOBILITY, ROOK_MOBILITY, SUPPORTED_PAWN,
};
use crate::eval::{PHASE_TOTAL, SCALE_NORMAL, Term, Terms, collect_terms, endgame_scale, phase};
use crate::types::{Color, Piece, SQUARES};

const DEFAULT_EPOCHS: usize = 2_000;
const REPORT_INTERVAL: usize = 50;
const LEARNING_RATE: f64 = 1.0;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPSILON: f64 = 1e-8;

type Weights = Vec<[f64; 2]>;

enum Layout {
    Scalar(Term),
    Table(Vec<Term>),
    PieceSquare,
}

impl Layout {
    fn terms(&self) -> Vec<Term> {
        match self {
            Self::Scalar(term) => vec![*term],
            Self::Table(terms) => terms.clone(),
            Self::PieceSquare => Piece::ALL
                .into_iter()
                .flat_map(piece_square_terms)
                .collect(),
        }
    }
}

fn piece_square_terms(piece: Piece) -> Vec<Term> {
    (0..SQUARES)
        .map(|square| Term::PieceSquare(piece, square))
        .collect()
}

fn indexed(len: usize, term: impl Fn(usize) -> Term) -> Layout {
    Layout::Table((0..len).map(term).collect())
}

fn per_piece(term: fn(Piece) -> Term) -> Layout {
    Layout::Table(Piece::ALL.into_iter().map(term).collect())
}

fn mobility(piece: Piece, len: usize) -> Layout {
    indexed(len, |count| Term::Mobility(piece, count))
}

fn parameter_tables() -> Vec<(&'static str, Layout)> {
    vec![
        ("MATERIAL", per_piece(Term::Material)),
        ("PIECE_SQUARE", Layout::PieceSquare),
        (
            "KNIGHT_MOBILITY",
            mobility(Piece::Knight, KNIGHT_MOBILITY.len()),
        ),
        (
            "BISHOP_MOBILITY",
            mobility(Piece::Bishop, BISHOP_MOBILITY.len()),
        ),
        ("ROOK_MOBILITY", mobility(Piece::Rook, ROOK_MOBILITY.len())),
        (
            "QUEEN_MOBILITY",
            mobility(Piece::Queen, QUEEN_MOBILITY.len()),
        ),
        ("PASSED_PAWN", indexed(PASSED_PAWN.len(), Term::PassedPawn)),
        (
            "SUPPORTED_PAWN",
            indexed(SUPPORTED_PAWN.len(), Term::SupportedPawn),
        ),
        (
            "PHALANX_PAWN",
            indexed(PHALANX_PAWN.len(), Term::PhalanxPawn),
        ),
        (
            "PASSED_PAWN_BLOCKED",
            indexed(PASSED_PAWN_BLOCKED.len(), Term::PassedPawnBlocked),
        ),
        ("KING_DANGER", indexed(KING_DANGER.len(), Term::KingDanger)),
        ("THREAT_BY_PAWN", per_piece(Term::ThreatByPawn)),
        ("THREAT_BY_MINOR", per_piece(Term::ThreatByMinor)),
        ("THREAT_BY_ROOK", per_piece(Term::ThreatByRook)),
        ("DOUBLED_PAWN", Layout::Scalar(Term::DoubledPawn)),
        ("ISOLATED_PAWN", Layout::Scalar(Term::IsolatedPawn)),
        ("BACKWARD_PAWN", Layout::Scalar(Term::BackwardPawn)),
        (
            "PASSED_OWN_KING_DISTANCE",
            Layout::Scalar(Term::PassedOwnKingDistance),
        ),
        (
            "PASSED_ENEMY_KING_DISTANCE",
            Layout::Scalar(Term::PassedEnemyKingDistance),
        ),
        ("BISHOP_PAIR", Layout::Scalar(Term::BishopPair)),
        ("ROOK_OPEN_FILE", Layout::Scalar(Term::RookOpenFile)),
        (
            "ROOK_SEMI_OPEN_FILE",
            Layout::Scalar(Term::RookSemiOpenFile),
        ),
        ("KNIGHT_OUTPOST", Layout::Scalar(Term::KnightOutpost)),
        ("BISHOP_OUTPOST", Layout::Scalar(Term::BishopOutpost)),
        ("PAWN_SHIELD", Layout::Scalar(Term::PawnShield)),
        ("HANGING", Layout::Scalar(Term::Hanging)),
        ("TEMPO", Layout::Scalar(Term::Tempo)),
    ]
}

fn all_terms() -> Vec<Term> {
    parameter_tables()
        .iter()
        .flat_map(|(_, layout)| layout.terms())
        .collect()
}

fn term_indices() -> HashMap<Term, usize> {
    all_terms()
        .into_iter()
        .enumerate()
        .map(|(index, term)| (term, index))
        .collect()
}

fn initial_weights() -> Weights {
    all_terms()
        .into_iter()
        .map(|term| {
            let weight = term.weight();
            [f64::from(weight.mg), f64::from(weight.eg)]
        })
        .collect()
}

struct Trace<'a> {
    counts: Vec<i32>,
    indices: &'a HashMap<Term, usize>,
}

impl<'a> Trace<'a> {
    fn new(indices: &'a HashMap<Term, usize>) -> Self {
        Self {
            counts: vec![0; indices.len()],
            indices,
        }
    }
}

impl Terms for Trace<'_> {
    fn add(&mut self, side: Color, term: Term, count: i32) {
        let signed = match side {
            Color::White => count,
            Color::Black => -count,
        };
        let index = *self
            .indices
            .get(&term)
            .unwrap_or_else(|| panic!("term {term:?} has no tunable parameter"));
        self.counts[index] += signed;
    }
}

struct Position {
    features: std::ops::Range<usize>,
    phase: f64,
    endgame_scale: f64,
    result: f64,
}

struct Dataset {
    positions: Vec<Position>,
    features: Vec<(u16, i16)>,
}

impl Dataset {
    fn new() -> Self {
        Self {
            positions: Vec::new(),
            features: Vec::new(),
        }
    }

    fn push(&mut self, board: &Board, result: f64, trace: &mut Trace) {
        trace.counts.fill(0);
        collect_terms(board, trace);
        let start = self.features.len();
        for (index, &count) in trace.counts.iter().enumerate() {
            if count != 0 {
                self.features.push((index as u16, count as i16));
            }
        }
        self.positions.push(Position {
            features: start..self.features.len(),
            phase: f64::from(phase(board)) / f64::from(PHASE_TOTAL),
            endgame_scale: f64::from(endgame_scale(board)) / f64::from(SCALE_NORMAL),
            result,
        });
    }
}

fn parse_result(line: &str) -> Option<f64> {
    if line.contains("1/2-1/2") {
        Some(0.5)
    } else if line.contains("1-0") {
        Some(1.0)
    } else if line.contains("0-1") {
        Some(0.0)
    } else {
        None
    }
}

fn load_dataset(path: &str) -> io::Result<Dataset> {
    let reader = BufReader::new(fs::File::open(path)?);
    let indices = term_indices();
    let mut dataset = Dataset::new();
    let mut trace = Trace::new(&indices);
    for line in reader.lines() {
        let line = line?;
        let Some(result) = parse_result(&line) else {
            continue;
        };
        let fen: Vec<&str> = line.split_whitespace().take(4).collect();
        let Ok(board) = format!("{} 0 1", fen.join(" ")).parse::<Board>() else {
            continue;
        };
        dataset.push(&board, result, &mut trace);
    }
    Ok(dataset)
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn phase_shares(position: &Position) -> (f64, f64) {
    (
        position.phase,
        (1.0 - position.phase) * position.endgame_scale,
    )
}

fn linear_eval(dataset: &Dataset, position: &Position, weights: &Weights) -> f64 {
    let (mut mg, mut eg) = (0.0, 0.0);
    for &(index, count) in &dataset.features[position.features.clone()] {
        let [weight_mg, weight_eg] = weights[usize::from(index)];
        mg += weight_mg * f64::from(count);
        eg += weight_eg * f64::from(count);
    }
    let (mg_share, eg_share) = phase_shares(position);
    mg * mg_share + eg * eg_share
}

fn parallel_chunks<T: Send>(dataset: &Dataset, work: impl Fn(&[Position]) -> T + Sync) -> Vec<T> {
    let threads = thread::available_parallelism().map_or(1, usize::from);
    let chunk = dataset.positions.len().div_ceil(threads).max(1);
    thread::scope(|scope| {
        let handles: Vec<_> = dataset
            .positions
            .chunks(chunk)
            .map(|positions| scope.spawn(|| work(positions)))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("tuner worker finished"))
            .collect()
    })
}

fn loss(dataset: &Dataset, weights: &Weights, scale: f64) -> f64 {
    let total: f64 = parallel_chunks(dataset, |positions| {
        positions
            .iter()
            .map(|position| {
                let predicted = sigmoid(scale * linear_eval(dataset, position, weights));
                (position.result - predicted).powi(2)
            })
            .sum::<f64>()
    })
    .into_iter()
    .sum();
    total / dataset.positions.len() as f64
}

fn gradient(dataset: &Dataset, weights: &Weights, scale: f64) -> Weights {
    let partials = parallel_chunks(dataset, |positions| {
        let mut partial = vec![[0.0; 2]; weights.len()];
        for position in positions {
            let predicted = sigmoid(scale * linear_eval(dataset, position, weights));
            let slope = (predicted - position.result) * predicted * (1.0 - predicted);
            let (mg_share, eg_share) = phase_shares(position);
            for &(index, count) in &dataset.features[position.features.clone()] {
                let entry = &mut partial[usize::from(index)];
                entry[0] += slope * mg_share * f64::from(count);
                entry[1] += slope * eg_share * f64::from(count);
            }
        }
        partial
    });
    let mut total = vec![[0.0; 2]; weights.len()];
    for partial in partials {
        for (sum, value) in total.iter_mut().zip(partial) {
            sum[0] += value[0];
            sum[1] += value[1];
        }
    }
    total
}

fn fit_scale(dataset: &Dataset, weights: &Weights) -> f64 {
    let (mut low, mut high) = (0.0001, 0.05);
    for _ in 0..40 {
        let left = low + (high - low) / 3.0;
        let right = high - (high - low) / 3.0;
        if loss(dataset, weights, left) < loss(dataset, weights, right) {
            high = right;
        } else {
            low = left;
        }
    }
    (low + high) / 2.0
}

pub fn run(arguments: &[String]) -> io::Result<()> {
    let [dataset_path, output_path, rest @ ..] = arguments else {
        eprintln!("usage: arabica tune <dataset.epd> <params.rs> [epochs]");
        return Ok(());
    };
    let epochs = rest
        .first()
        .and_then(|text| text.parse().ok())
        .unwrap_or(DEFAULT_EPOCHS);

    let dataset = load_dataset(dataset_path)?;
    println!("positions {}", dataset.positions.len());
    let mut weights = initial_weights();
    let scale = fit_scale(&dataset, &weights);
    println!(
        "scale {scale:.6} loss {:.6}",
        loss(&dataset, &weights, scale)
    );

    let mut first_moment = vec![[0.0; 2]; weights.len()];
    let mut second_moment = vec![[0.0; 2]; weights.len()];
    for epoch in 1..=epochs {
        let gradient = gradient(&dataset, &weights, scale);
        let correction1 = 1.0 - BETA1.powi(epoch as i32);
        let correction2 = 1.0 - BETA2.powi(epoch as i32);
        for index in 0..weights.len() {
            for phase in 0..2 {
                let grad = gradient[index][phase];
                let first = &mut first_moment[index][phase];
                let second = &mut second_moment[index][phase];
                *first = BETA1 * *first + (1.0 - BETA1) * grad;
                *second = BETA2 * *second + (1.0 - BETA2) * grad * grad;
                let step = LEARNING_RATE * (*first / correction1)
                    / ((*second / correction2).sqrt() + EPSILON);
                weights[index][phase] -= step;
            }
        }
        if epoch % REPORT_INTERVAL == 0 || epoch == epochs {
            println!("epoch {epoch} loss {:.6}", loss(&dataset, &weights, scale));
            fs::write(output_path, render_params(&weights))?;
        }
    }
    Ok(())
}

fn render_score(weights: &Weights, indices: &HashMap<Term, usize>, term: Term) -> String {
    let [mg, eg] = weights[indices[&term]];
    format!("s({:4}, {:4})", mg.round() as i32, eg.round() as i32)
}

fn render_rows(output: &mut String, indent: &str, scores: &[String]) {
    for row in scores.chunks(8) {
        writeln!(output, "{indent}{},", row.join(", ")).unwrap();
    }
}

fn render_params(weights: &Weights) -> String {
    let indices = term_indices();
    let render = |term: Term| render_score(weights, &indices, term);
    let mut output = String::from("use crate::eval::{Score, s};\n");
    for (name, layout) in parameter_tables() {
        match layout {
            Layout::Scalar(term) => {
                let [mg, eg] = weights[indices[&term]];
                let (mg, eg) = (mg.round() as i32, eg.round() as i32);
                writeln!(output, "\npub const {name}: Score = s({mg}, {eg});").unwrap();
            }
            Layout::Table(terms) => {
                let scores: Vec<String> = terms.into_iter().map(render).collect();
                writeln!(output, "\n#[rustfmt::skip]").unwrap();
                writeln!(output, "pub const {name}: [Score; {}] = [", scores.len()).unwrap();
                render_rows(&mut output, "    ", &scores);
                writeln!(output, "];").unwrap();
            }
            Layout::PieceSquare => {
                writeln!(output, "\n#[rustfmt::skip]").unwrap();
                writeln!(output, "pub const {name}: [[Score; 64]; 6] = [").unwrap();
                for piece in Piece::ALL {
                    let scores: Vec<String> =
                        piece_square_terms(piece).into_iter().map(render).collect();
                    writeln!(output, "    [").unwrap();
                    render_rows(&mut output, "        ", &scores);
                    writeln!(output, "    ],").unwrap();
                }
                writeln!(output, "];").unwrap();
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::testing::board;
    use crate::eval::PawnCache;

    #[test]
    fn every_term_has_a_unique_parameter() {
        let terms = all_terms();
        let indices = term_indices();
        assert_eq!(indices.len(), terms.len(), "duplicate terms");
        for (expected, term) in terms.into_iter().enumerate() {
            assert_eq!(indices[&term], expected, "term {term:?}");
        }
    }

    #[test]
    fn rendered_params_round_trip_current_weights() {
        let rendered = render_params(&initial_weights());
        let current = include_str!("eval/params.rs");
        let normalized = |text: &str| -> Vec<String> {
            text.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(String::from)
                .collect()
        };
        let mut rendered_lines = normalized(&rendered);
        let mut current_lines = normalized(current);
        rendered_lines.sort();
        current_lines.sort();
        assert_eq!(rendered_lines, current_lines);
    }

    #[test]
    fn linear_eval_matches_engine_eval() {
        let cases = [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                1.0,
            ),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                1.0,
            ),
            ("8/8/1p1k4/1P6/8/3p3P/1r4P1/5K2 w - - 0 1", 1.0),
            ("6k1/5ppp/8/1P6/8/2P5/2P2PPP/3R2K1 b - - 0 1", 1.0),
            ("6k1/5p1p/4N3/8/8/8/8/4K1Q1 w - - 0 1", 1.0),
            ("4kb2/p7/8/8/8/8/PP6/3BK3 w - - 0 1", 2.0),
            ("4kn2/8/8/8/8/8/8/R3K3 b - - 0 1", 2.0),
        ];
        let weights = initial_weights();
        let indices = term_indices();

        for (fen, tolerance) in cases {
            let board = board(fen);
            let mut dataset = Dataset::new();
            dataset.push(&board, 0.5, &mut Trace::new(&indices));
            let white_eval = crate::eval::evaluate(&board, &mut PawnCache::default())
                * match board.state.active_color {
                    Color::White => 1,
                    Color::Black => -1,
                };
            let linear = linear_eval(&dataset, &dataset.positions[0], &weights);
            assert!(
                (linear - f64::from(white_eval)).abs() <= tolerance,
                "fen {fen:?}: linear {linear}, engine {white_eval}"
            );
        }
    }
}
