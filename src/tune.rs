use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::thread;

use crate::board::Board;
use crate::eval::params::{
    BISHOP_MOBILITY, KING_ZONE_ATTACKS, KNIGHT_MOBILITY, MATERIAL, PASSED_PAWN, PIECE_SQUARE,
    QUEEN_MOBILITY, ROOK_MOBILITY,
};
use crate::eval::{PHASE_TOTAL, Term, Terms, collect_terms, phase};
use crate::types::{Color, Piece, SQUARES};

const DEFAULT_EPOCHS: usize = 2_000;
const REPORT_INTERVAL: usize = 50;
const LEARNING_RATE: f64 = 1.0;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPSILON: f64 = 1e-8;

const PIECE_SQUARE_OFFSET: usize = MATERIAL.len();
const KNIGHT_MOBILITY_OFFSET: usize = PIECE_SQUARE_OFFSET + PIECE_SQUARE.len() * SQUARES;
const BISHOP_MOBILITY_OFFSET: usize = KNIGHT_MOBILITY_OFFSET + KNIGHT_MOBILITY.len();
const ROOK_MOBILITY_OFFSET: usize = BISHOP_MOBILITY_OFFSET + BISHOP_MOBILITY.len();
const QUEEN_MOBILITY_OFFSET: usize = ROOK_MOBILITY_OFFSET + ROOK_MOBILITY.len();
const PASSED_PAWN_OFFSET: usize = QUEEN_MOBILITY_OFFSET + QUEEN_MOBILITY.len();
const DOUBLED_PAWN_INDEX: usize = PASSED_PAWN_OFFSET + PASSED_PAWN.len();
const ISOLATED_PAWN_INDEX: usize = DOUBLED_PAWN_INDEX + 1;
const BISHOP_PAIR_INDEX: usize = ISOLATED_PAWN_INDEX + 1;
const ROOK_OPEN_FILE_INDEX: usize = BISHOP_PAIR_INDEX + 1;
const ROOK_SEMI_OPEN_FILE_INDEX: usize = ROOK_OPEN_FILE_INDEX + 1;
const PAWN_SHIELD_INDEX: usize = ROOK_SEMI_OPEN_FILE_INDEX + 1;
const KING_ZONE_ATTACKS_OFFSET: usize = PAWN_SHIELD_INDEX + 1;
const PARAM_COUNT: usize = KING_ZONE_ATTACKS_OFFSET + KING_ZONE_ATTACKS.len();

type Weights = Vec<[f64; 2]>;

fn term_index(term: Term) -> usize {
    match term {
        Term::Material(piece) => piece.index(),
        Term::PieceSquare(piece, square) => PIECE_SQUARE_OFFSET + piece.index() * SQUARES + square,
        Term::Mobility(Piece::Knight, count) => KNIGHT_MOBILITY_OFFSET + count,
        Term::Mobility(Piece::Bishop, count) => BISHOP_MOBILITY_OFFSET + count,
        Term::Mobility(Piece::Rook, count) => ROOK_MOBILITY_OFFSET + count,
        Term::Mobility(_, count) => QUEEN_MOBILITY_OFFSET + count,
        Term::PassedPawn(rank) => PASSED_PAWN_OFFSET + rank,
        Term::DoubledPawn => DOUBLED_PAWN_INDEX,
        Term::IsolatedPawn => ISOLATED_PAWN_INDEX,
        Term::BishopPair => BISHOP_PAIR_INDEX,
        Term::RookOpenFile => ROOK_OPEN_FILE_INDEX,
        Term::RookSemiOpenFile => ROOK_SEMI_OPEN_FILE_INDEX,
        Term::PawnShield => PAWN_SHIELD_INDEX,
        Term::KingZoneAttacks(attacker) => KING_ZONE_ATTACKS_OFFSET + attacker,
    }
}

fn all_terms() -> Vec<Term> {
    let mobility =
        |piece: Piece, len: usize| (0..len).map(move |count| Term::Mobility(piece, count));
    Piece::ALL
        .into_iter()
        .map(Term::Material)
        .chain(
            Piece::ALL
                .into_iter()
                .flat_map(|piece| (0..SQUARES).map(move |square| Term::PieceSquare(piece, square))),
        )
        .chain(mobility(Piece::Knight, KNIGHT_MOBILITY.len()))
        .chain(mobility(Piece::Bishop, BISHOP_MOBILITY.len()))
        .chain(mobility(Piece::Rook, ROOK_MOBILITY.len()))
        .chain(mobility(Piece::Queen, QUEEN_MOBILITY.len()))
        .chain((0..PASSED_PAWN.len()).map(Term::PassedPawn))
        .chain([
            Term::DoubledPawn,
            Term::IsolatedPawn,
            Term::BishopPair,
            Term::RookOpenFile,
            Term::RookSemiOpenFile,
            Term::PawnShield,
        ])
        .chain((0..KING_ZONE_ATTACKS.len()).map(Term::KingZoneAttacks))
        .collect()
}

struct Trace(Vec<i32>);

impl Terms for Trace {
    fn add(&mut self, side: Color, term: Term, count: i32) {
        let signed = match side {
            Color::White => count,
            Color::Black => -count,
        };
        self.0[term_index(term)] += signed;
    }
}

struct Position {
    features: std::ops::Range<usize>,
    phase: f64,
    result: f64,
}

struct Dataset {
    positions: Vec<Position>,
    features: Vec<(u16, i16)>,
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
    let mut dataset = Dataset {
        positions: Vec::new(),
        features: Vec::new(),
    };
    let mut trace = Trace(vec![0; PARAM_COUNT]);
    for line in reader.lines() {
        let line = line?;
        let Some(result) = parse_result(&line) else {
            continue;
        };
        let fen: Vec<&str> = line.split_whitespace().take(4).collect();
        let Ok(board) = format!("{} 0 1", fen.join(" ")).parse::<Board>() else {
            continue;
        };
        trace.0.fill(0);
        collect_terms(&board, &mut trace);
        let start = dataset.features.len();
        for (index, &count) in trace.0.iter().enumerate() {
            if count != 0 {
                dataset.features.push((index as u16, count as i16));
            }
        }
        dataset.positions.push(Position {
            features: start..dataset.features.len(),
            phase: f64::from(phase(&board)) / f64::from(PHASE_TOTAL),
            result,
        });
    }
    Ok(dataset)
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn linear_eval(dataset: &Dataset, position: &Position, weights: &Weights) -> f64 {
    let (mut mg, mut eg) = (0.0, 0.0);
    for &(index, count) in &dataset.features[position.features.clone()] {
        let [weight_mg, weight_eg] = weights[usize::from(index)];
        mg += weight_mg * f64::from(count);
        eg += weight_eg * f64::from(count);
    }
    mg * position.phase + eg * (1.0 - position.phase)
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
        let mut partial = vec![[0.0; 2]; PARAM_COUNT];
        for position in positions {
            let predicted = sigmoid(scale * linear_eval(dataset, position, weights));
            let slope = (predicted - position.result) * predicted * (1.0 - predicted);
            let (mg_share, eg_share) = (slope * position.phase, slope * (1.0 - position.phase));
            for &(index, count) in &dataset.features[position.features.clone()] {
                let entry = &mut partial[usize::from(index)];
                entry[0] += mg_share * f64::from(count);
                entry[1] += eg_share * f64::from(count);
            }
        }
        partial
    });
    let mut total = vec![[0.0; 2]; PARAM_COUNT];
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
    let mut weights: Weights = all_terms()
        .into_iter()
        .map(|term| {
            let weight = term.weight();
            [f64::from(weight.mg), f64::from(weight.eg)]
        })
        .collect();
    let scale = fit_scale(&dataset, &weights);
    println!(
        "scale {scale:.6} loss {:.6}",
        loss(&dataset, &weights, scale)
    );

    let mut first_moment = vec![[0.0; 2]; PARAM_COUNT];
    let mut second_moment = vec![[0.0; 2]; PARAM_COUNT];
    for epoch in 1..=epochs {
        let gradient = gradient(&dataset, &weights, scale);
        let correction1 = 1.0 - BETA1.powi(epoch as i32);
        let correction2 = 1.0 - BETA2.powi(epoch as i32);
        for index in 0..PARAM_COUNT {
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

fn rounded(weights: &Weights, term: Term) -> (i32, i32) {
    let [mg, eg] = weights[term_index(term)];
    (mg.round() as i32, eg.round() as i32)
}

fn render_score(weights: &Weights, term: Term) -> String {
    let (mg, eg) = rounded(weights, term);
    format!("s({mg:4}, {eg:4})")
}

fn render_array(output: &mut String, name: &str, terms: &[Term], weights: &Weights) {
    let rendered: Vec<String> = terms
        .iter()
        .map(|&term| render_score(weights, term))
        .collect();
    writeln!(output, "\n#[rustfmt::skip]").unwrap();
    writeln!(output, "pub const {name}: [Score; {}] = [", terms.len()).unwrap();
    for row in rendered.chunks(8) {
        writeln!(output, "    {},", row.join(", ")).unwrap();
    }
    writeln!(output, "];").unwrap();
}

fn render_params(weights: &Weights) -> String {
    let mut output = String::from("use crate::eval::{Score, s};\n");
    let material: Vec<Term> = Piece::ALL.into_iter().map(Term::Material).collect();
    render_array(&mut output, "MATERIAL", &material, weights);

    writeln!(output, "\n#[rustfmt::skip]").unwrap();
    writeln!(output, "pub const PIECE_SQUARE: [[Score; 64]; 6] = [").unwrap();
    for piece in Piece::ALL {
        writeln!(output, "    [").unwrap();
        for rank in 0..8 {
            let row: Vec<String> = (0..8)
                .map(|file| render_score(weights, Term::PieceSquare(piece, rank * 8 + file)))
                .collect();
            writeln!(output, "        {},", row.join(", ")).unwrap();
        }
        writeln!(output, "    ],").unwrap();
    }
    writeln!(output, "];").unwrap();

    let mobility = |piece: Piece, len: usize| -> Vec<Term> {
        (0..len).map(|count| Term::Mobility(piece, count)).collect()
    };
    render_array(
        &mut output,
        "KNIGHT_MOBILITY",
        &mobility(Piece::Knight, KNIGHT_MOBILITY.len()),
        weights,
    );
    render_array(
        &mut output,
        "BISHOP_MOBILITY",
        &mobility(Piece::Bishop, BISHOP_MOBILITY.len()),
        weights,
    );
    render_array(
        &mut output,
        "ROOK_MOBILITY",
        &mobility(Piece::Rook, ROOK_MOBILITY.len()),
        weights,
    );
    render_array(
        &mut output,
        "QUEEN_MOBILITY",
        &mobility(Piece::Queen, QUEEN_MOBILITY.len()),
        weights,
    );
    let passed: Vec<Term> = (0..PASSED_PAWN.len()).map(Term::PassedPawn).collect();
    render_array(&mut output, "PASSED_PAWN", &passed, weights);

    for (name, term) in [
        ("DOUBLED_PAWN", Term::DoubledPawn),
        ("ISOLATED_PAWN", Term::IsolatedPawn),
        ("BISHOP_PAIR", Term::BishopPair),
        ("ROOK_OPEN_FILE", Term::RookOpenFile),
        ("ROOK_SEMI_OPEN_FILE", Term::RookSemiOpenFile),
        ("PAWN_SHIELD", Term::PawnShield),
    ] {
        let (mg, eg) = rounded(weights, term);
        writeln!(output, "\npub const {name}: Score = s({mg}, {eg});").unwrap();
    }
    let king_zone: Vec<Term> = (0..KING_ZONE_ATTACKS.len())
        .map(Term::KingZoneAttacks)
        .collect();
    render_array(&mut output, "KING_ZONE_ATTACKS", &king_zone, weights);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_indices_are_dense_and_ordered() {
        let terms = all_terms();
        assert_eq!(terms.len(), PARAM_COUNT);
        for (expected, term) in terms.into_iter().enumerate() {
            assert_eq!(term_index(term), expected, "term {term:?}");
        }
    }

    #[test]
    fn linear_eval_matches_engine_eval() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/8/1p1k4/1P6/8/3p3P/1r4P1/5K2 w - - 0 1",
            "6k1/5ppp/8/1P6/8/2P5/2P2PPP/3R2K1 b - - 0 1",
        ];
        let weights: Weights = all_terms()
            .into_iter()
            .map(|term| [f64::from(term.weight().mg), f64::from(term.weight().eg)])
            .collect();

        for fen in fens {
            let board: Board = fen
                .parse()
                .unwrap_or_else(|error| panic!("fen {fen:?} rejected: {error}"));
            let mut trace = Trace(vec![0; PARAM_COUNT]);
            collect_terms(&board, &mut trace);
            let features: Vec<(u16, i16)> = trace
                .0
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count != 0)
                .map(|(index, &count)| (index as u16, count as i16))
                .collect();
            let dataset = Dataset {
                positions: vec![Position {
                    features: 0..features.len(),
                    phase: f64::from(phase(&board)) / f64::from(PHASE_TOTAL),
                    result: 0.5,
                }],
                features,
            };
            let white_eval = crate::eval::evaluate(&board)
                * match board.state.active_color {
                    Color::White => 1,
                    Color::Black => -1,
                };
            let linear = linear_eval(&dataset, &dataset.positions[0], &weights);
            assert!(
                (linear - f64::from(white_eval)).abs() <= 1.0,
                "fen {fen:?}: linear {linear}, engine {white_eval}"
            );
        }
    }
}
