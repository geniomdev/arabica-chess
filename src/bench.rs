use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use crate::history::History;
use crate::search::{SearchLimits, Searcher};
use crate::tt::{DEFAULT_HASH_MB, TranspositionTable};
use crate::uci::parse_position;

const DEFAULT_BENCH_DEPTH: u8 = 13;

const BENCH_POSITIONS: [&str; 49] = [
    "fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
    "fen 4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    "fen rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14 moves d4e6",
    "fen r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14 moves g2g4",
    "fen r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
    "fen r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
    "fen r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
    "fen 4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
    "fen 2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
    "fen r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
    "fen 3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
    "fen r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18",
    "fen 4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
    "fen 3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26",
    "fen 6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
    "fen 3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
    "fen 2K5/p7/7P/5pR1/8/5k2/r7/8 w - - 0 1 moves g5g6 f3e3 g6g5 e3f3",
    "fen 8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
    "fen 7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
    "fen 8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1",
    "fen 8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1",
    "fen 8/pp2r1k1/2p1p3/3pP2p/1P1P1P1P/P5KR/8/8 w - - 0 1",
    "fen 8/3p4/p1bk3p/Pp6/1Kp1PpPp/2P2P1P/2P5/5B2 b - - 0 1",
    "fen 5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
    "fen 6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1",
    "fen 1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
    "fen 6k1/4pp1p/3p2p1/P1pPb3/R7/1r2P1PP/3B1P2/6K1 w - - 0 1",
    "fen 8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1",
    "fen 5rk1/q6p/2p3bR/1pPp1rP1/1P1Pp3/P3B1Q1/1K3P2/R7 w - - 93 90",
    "fen 4rrk1/1p1nq3/p7/2p1P1pp/3P2bp/3Q1Bn1/PPPB4/1K2R1NR w - - 40 21",
    "fen r3k2r/3nnpbp/q2pp1p1/p7/Pp1PPPP1/4BNN1/1P5P/R2Q1RK1 w kq - 0 16",
    "fen 3Qb1k1/1r2ppb1/pN1n2q1/Pp1Pp1Pr/4P2p/4BP2/4B1R1/1R5K b - - 11 40",
    "fen 4k3/3q1r2/1N2r1b1/3ppN2/2nPP3/1B1R2n1/2R1Q3/3K4 w - - 5 1",
    "fen 1r6/1P4bk/3qr1p1/N6p/3pp2P/6R1/3Q1PP1/1R4K1 w - - 1 42",
    "fen k7/2n1n3/1nbNbn2/2NbRBn1/1nbRQR2/2NBRBN1/3N1N2/7K w - - 0 1",
    "fen K7/8/8/BNQNQNB1/N5N1/R1Q1q2r/n5n1/bnqnqnbk w - - 0 1",
    "fen 8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1",
    "fen 8/8/8/5N2/8/p7/8/2NK3k w - - 0 1",
    "fen 8/3k4/8/8/8/4B3/4KB2/2B5 w - - 0 1",
    "fen 8/8/1P6/5pr1/8/4R3/7k/2K5 w - - 0 1",
    "fen 8/2p4P/8/kr6/6R1/8/8/1K6 w - - 0 1",
    "fen 8/8/3P3k/8/1p6/8/1P6/1K3n2 b - - 0 1",
    "fen 8/R7/2q5/8/6k1/8/1P5p/K6R w - - 0 124",
    "fen 6k1/3b3r/1p1p4/p1n2p2/1PPNpP1q/P3Q1p1/1R1RB1P1/5K2 b - - 0 1",
    "fen r2r1n2/pp2bk2/2p1p2p/3q4/3PN1QP/2P3R1/P4PP1/5RK1 w - - 0 1",
    "fen 8/8/8/8/8/6k1/6p1/6K1 w - -",
    "fen 7k/7P/6K1/8/3B4/8/8/8 b - -",
];

pub fn run(arguments: &[&str]) {
    let depth = match arguments.first() {
        Some(text) => match text.parse() {
            Ok(depth) => depth,
            Err(_) => {
                println!("info string invalid bench depth: {text}");
                return;
            }
        },
        None => DEFAULT_BENCH_DEPTH,
    };

    let limits = SearchLimits {
        depth: Some(depth),
        ..SearchLimits::default()
    };
    let mut table = TranspositionTable::new(DEFAULT_HASH_MB);
    let mut history = History::default();
    let mut total_nodes = 0;
    let mut total_time = Duration::ZERO;

    println!(
        "{:>2}  {:>10}  {:>7}  {:>8}  {}",
        "#", "nodes", "time ms", "nps", "bestmove"
    );
    println!("{:-<2}  {:-<10}  {:-<7}  {:-<8}  {:-<8}", "", "", "", "", "");
    for (index, setup) in BENCH_POSITIONS.iter().enumerate() {
        let tokens: Vec<&str> = setup.split_whitespace().collect();
        let mut board = parse_position(&tokens).expect("bench position is valid");
        table.clear();
        history.clear();

        let mut nodes = 0;
        let started = Instant::now();
        let mut searcher = Searcher::new(limits, Arc::new(AtomicBool::new(false)), table)
            .with_history(history);
        let (best_move, _) = searcher.search(&mut board, |iteration| nodes = iteration.nodes);
        let elapsed = started.elapsed();
        total_time += elapsed;
        (table, history) = searcher.into_parts();

        total_nodes += nodes;
        println!(
            "{:>2}  {nodes:>10}  {:>7}  {:>8}  {best_move}",
            index + 1,
            elapsed.as_millis(),
            nodes_per_second(nodes, elapsed),
        );
    }

    let nps = nodes_per_second(total_nodes, total_time);
    println!();
    println!("Total time (ms) : {}", total_time.as_millis());
    println!("Nodes searched  : {total_nodes}");
    println!("Nodes/second    : {nps}");
}

fn nodes_per_second(nodes: u64, elapsed: Duration) -> u64 {
    (nodes as f64 / elapsed.as_secs_f64().max(f64::EPSILON)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_positions_are_valid() {
        for setup in BENCH_POSITIONS {
            let tokens: Vec<&str> = setup.split_whitespace().collect();
            assert!(parse_position(&tokens).is_ok(), "position {setup:?}");
        }
    }
}
