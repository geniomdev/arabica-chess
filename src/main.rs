mod attacks;
mod board;
mod eval;
mod search;
mod types;

use board::{Board, START_POSITION};
use search::{SearchLimits, Searcher, uci_score};

const DEFAULT_DEPTH: u8 = 6;

fn main() {
    attacks::init();

    let fen = std::env::args()
        .nth(1)
        .unwrap_or_else(|| START_POSITION.to_string());
    let depth = std::env::args().nth(2).map_or(DEFAULT_DEPTH, |depth| {
        depth.parse().expect("depth is a number")
    });

    let mut board: Board = fen.parse().expect("valid position");
    println!("{board}\n");

    let limits = SearchLimits {
        depth: Some(depth),
        time: None,
    };
    let (best_move, _) = Searcher::new(limits).search(&mut board, |iteration| {
        let pv: Vec<String> = iteration.pv.iter().map(ToString::to_string).collect();
        println!(
            "info depth {} score {} nodes {} time {} pv {}",
            iteration.depth,
            uci_score(iteration.score),
            iteration.nodes,
            iteration.elapsed.as_millis(),
            pv.join(" ")
        );
    });
    println!("bestmove {best_move}");
}
