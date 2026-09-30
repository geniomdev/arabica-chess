use std::fmt;
use std::io::{self, BufRead};
use std::mem;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::board::{Board, FenError, MoveList, START_POSITION};
use crate::search::{Iteration, SearchLimits, Searcher, uci_score};
use crate::tt::{DEFAULT_HASH_MB, MAX_HASH_MB, MIN_HASH_MB, TranspositionTable};
use crate::types::{Color, EverySide};

const ENGINE_NAME: &str = "Arabica";
const ENGINE_AUTHOR: &str = "Geniomdev";
const MOVE_OVERHEAD: Duration = Duration::from_millis(50);
const DEFAULT_MOVES_TO_GO: u32 = 30;

pub fn run() {
    let mut engine = Engine::new();
    for line in io::stdin().lock().lines() {
        let Ok(line) = line else {
            break;
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if engine.execute(&tokens) == Flow::Quit {
            break;
        }
    }
    engine.stop_search();
}

#[derive(PartialEq, Eq)]
enum Flow {
    Continue,
    Quit,
}

struct Engine {
    board: Board,
    table: TranspositionTable,
    search: Option<RunningSearch>,
}

impl Engine {
    fn new() -> Self {
        Self {
            board: start_position(),
            table: TranspositionTable::new(DEFAULT_HASH_MB),
            search: None,
        }
    }

    fn execute(&mut self, tokens: &[&str]) -> Flow {
        match tokens {
            ["uci", ..] => {
                println!("id name {ENGINE_NAME} {}", env!("CARGO_PKG_VERSION"));
                println!("id author {ENGINE_AUTHOR}");
                println!(
                    "option name Hash type spin default {DEFAULT_HASH_MB} min {MIN_HASH_MB} max {MAX_HASH_MB}"
                );
                println!("uciok");
            }
            ["isready", ..] => println!("readyok"),
            ["ucinewgame", ..] => {
                self.stop_search();
                self.board = start_position();
                self.table.clear();
            }
            ["setoption", arguments @ ..] => {
                self.stop_search();
                match parse_hash_option(arguments) {
                    Some(megabytes) => self.table = TranspositionTable::new(megabytes),
                    None => println!("info string unsupported option: {}", arguments.join(" ")),
                }
            }
            ["position", arguments @ ..] => {
                self.stop_search();
                match parse_position(arguments) {
                    Ok(board) => self.board = board,
                    Err(error) => println!("info string {error}"),
                }
            }
            ["go", arguments @ ..] => {
                self.stop_search();
                self.go(parse_go(arguments));
            }
            ["stop", ..] => self.stop_search(),
            ["d", ..] => println!(
                "{}\n\nFen: {}\nKey: {:016X}",
                self.board,
                self.board.to_fen(),
                self.board.state.zobrist_key
            ),
            ["quit", ..] => return Flow::Quit,
            _ => {}
        }
        Flow::Continue
    }

    fn go(&mut self, params: GoParams) {
        if let Some(depth) = params.perft {
            print_perft_divide(&mut self.board, depth);
            return;
        }
        let limits = params.limits(self.board.state.active_color);
        self.search = Some(RunningSearch::start(
            self.board.clone(),
            limits,
            mem::take(&mut self.table),
            params.infinite,
        ));
    }

    fn stop_search(&mut self) {
        if let Some(search) = self.search.take() {
            self.table = search.stop();
        }
    }
}

struct RunningSearch {
    stop_signal: Arc<AtomicBool>,
    thread: JoinHandle<TranspositionTable>,
}

impl RunningSearch {
    fn start(
        mut board: Board,
        limits: SearchLimits,
        table: TranspositionTable,
        hold_until_stopped: bool,
    ) -> Self {
        let stop_signal = Arc::new(AtomicBool::new(false));
        let searcher_signal = Arc::clone(&stop_signal);
        let thread = thread::spawn(move || {
            let mut searcher = Searcher::new(limits, Arc::clone(&searcher_signal), table);
            let (best_move, _) = searcher.search(&mut board, print_iteration);
            if hold_until_stopped {
                while !searcher_signal.load(Ordering::Acquire) {
                    thread::park();
                }
            }
            println!("bestmove {best_move}");
            searcher.into_table()
        });
        Self {
            stop_signal,
            thread,
        }
    }

    fn stop(self) -> TranspositionTable {
        self.stop_signal.store(true, Ordering::Release);
        self.thread.thread().unpark();
        self.thread.join().expect("search thread finished cleanly")
    }
}

fn print_iteration(iteration: &Iteration) {
    let pv: String = iteration
        .pv
        .iter()
        .map(|played| format!(" {played}"))
        .collect();
    let pv_section = if pv.is_empty() {
        pv
    } else {
        format!(" pv{pv}")
    };
    println!(
        "info depth {} score {} nodes {} time {} hashfull {}{pv_section}",
        iteration.depth,
        uci_score(iteration.score),
        iteration.nodes,
        iteration.elapsed.as_millis(),
        iteration.hashfull,
    );
}

fn print_perft_divide(board: &mut Board, depth: u32) {
    let mut moves = MoveList::new();
    board.generate_pseudo_legal(&mut moves);
    let mut total = 0;
    for &candidate in moves.as_slice() {
        if board.make_move(candidate) {
            let nodes = board.perft(depth.saturating_sub(1));
            board.unmake_move();
            println!("{candidate}: {nodes}");
            total += nodes;
        }
    }
    println!("\nNodes searched: {total}");
}

fn start_position() -> Board {
    START_POSITION.parse().expect("start position is valid FEN")
}

#[derive(Debug, PartialEq, Eq)]
enum PositionError {
    Setup(String),
    Fen(FenError),
    IllegalMove(String),
}

impl fmt::Display for PositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Setup(text) => write!(f, "invalid position setup: {text}"),
            Self::Fen(error) => write!(f, "invalid fen: {error}"),
            Self::IllegalMove(text) => write!(f, "illegal move: {text}"),
        }
    }
}

impl From<FenError> for PositionError {
    fn from(error: FenError) -> Self {
        Self::Fen(error)
    }
}

fn parse_position(arguments: &[&str]) -> Result<Board, PositionError> {
    let moves_start = arguments
        .iter()
        .position(|&token| token == "moves")
        .unwrap_or(arguments.len());
    let (setup, moves) = arguments.split_at(moves_start);
    let mut board: Board = match setup {
        ["startpos"] => start_position(),
        ["fen", fields @ ..] => fields.join(" ").parse()?,
        _ => return Err(PositionError::Setup(setup.join(" "))),
    };
    for &text in moves.iter().skip(1) {
        if !play_uci_move(&mut board, text) {
            return Err(PositionError::IllegalMove(text.to_string()));
        }
    }
    Ok(board)
}

fn play_uci_move(board: &mut Board, text: &str) -> bool {
    let mut moves = MoveList::new();
    board.generate_pseudo_legal(&mut moves);
    moves
        .as_slice()
        .iter()
        .find(|candidate| candidate.to_string() == text)
        .is_some_and(|&candidate| board.make_move(candidate))
}

#[derive(Debug, Default, PartialEq, Eq)]
struct GoParams {
    depth: Option<u8>,
    move_time: Option<Duration>,
    remaining: EverySide<Option<Duration>>,
    increment: EverySide<Option<Duration>>,
    moves_to_go: Option<u32>,
    infinite: bool,
    perft: Option<u32>,
}

impl GoParams {
    fn limits(&self, side: Color) -> SearchLimits {
        let mut limits = SearchLimits {
            depth: self.depth,
            ..SearchLimits::default()
        };
        if self.infinite {
            return limits;
        }
        if let Some(move_time) = self.move_time {
            limits.time = Some(move_time.saturating_sub(MOVE_OVERHEAD));
        } else if let Some(remaining) = self.remaining[side] {
            let increment = self.increment[side].unwrap_or_default();
            let budget = clock_budget(remaining, increment, self.moves_to_go);
            limits.time = Some(budget.hard);
            limits.soft_time = Some(budget.soft);
        }
        limits
    }
}

struct TimeBudget {
    soft: Duration,
    hard: Duration,
}

fn clock_budget(remaining: Duration, increment: Duration, moves_to_go: Option<u32>) -> TimeBudget {
    let available = remaining.saturating_sub(MOVE_OVERHEAD);
    let moves_left = moves_to_go.unwrap_or(DEFAULT_MOVES_TO_GO).max(1);
    let share = remaining / moves_left + increment * 3 / 4;
    let hard = share.min(available);
    TimeBudget {
        soft: hard / 2,
        hard,
    }
}

fn parse_hash_option(arguments: &[&str]) -> Option<usize> {
    match arguments {
        ["name", name, "value", value] if name.eq_ignore_ascii_case("hash") => value.parse().ok(),
        _ => None,
    }
}

fn parse_go(arguments: &[&str]) -> GoParams {
    let mut params = GoParams::default();
    let mut tokens = arguments.iter().copied();
    while let Some(key) = tokens.next() {
        match key {
            "infinite" => params.infinite = true,
            "depth" => params.depth = next_number(&mut tokens),
            "movetime" => params.move_time = next_millis(&mut tokens),
            "wtime" => params.remaining[Color::White] = next_millis(&mut tokens),
            "btime" => params.remaining[Color::Black] = next_millis(&mut tokens),
            "winc" => params.increment[Color::White] = next_millis(&mut tokens),
            "binc" => params.increment[Color::Black] = next_millis(&mut tokens),
            "movestogo" => params.moves_to_go = next_number(&mut tokens),
            "perft" => params.perft = next_number(&mut tokens),
            _ => {}
        }
    }
    params
}

fn next_number<'a, T: FromStr>(tokens: &mut impl Iterator<Item = &'a str>) -> Option<T> {
    tokens.next()?.parse().ok()
}

fn next_millis<'a>(tokens: &mut impl Iterator<Item = &'a str>) -> Option<Duration> {
    let millis: i64 = next_number(tokens)?;
    Some(Duration::from_millis(millis.max(0).unsigned_abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::KIWIPETE;

    fn tokens(command: &str) -> Vec<&str> {
        command.split_whitespace().collect()
    }

    fn per_side(white: Option<u64>, black: Option<u64>) -> EverySide<Option<Duration>> {
        let mut sides = EverySide::default();
        sides[Color::White] = white.map(Duration::from_millis);
        sides[Color::Black] = black.map(Duration::from_millis);
        sides
    }

    fn millis(value: u64) -> Option<Duration> {
        Some(Duration::from_millis(value))
    }

    #[test]
    fn parses_position_commands() {
        let kiwipete_moves = format!("fen {KIWIPETE} moves e1g1");
        let cases: [(&str, Result<&str, PositionError>); 9] = [
            ("startpos", Ok(START_POSITION)),
            (
                "startpos moves e2e4 e7e5",
                Ok("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
            ),
            (
                "startpos moves e2e4 a7a6 e4e5 d7d5 e5d6",
                Ok("rnbqkbnr/1pp1pppp/p2P4/8/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3"),
            ),
            (
                &kiwipete_moves,
                Ok("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R4RK1 b kq - 1 1"),
            ),
            (
                "fen 8/P6k/8/8/8/8/8/K7 w - - 0 1 moves a7a8q",
                Ok("Q7/7k/8/8/8/8/8/K7 b - - 0 1"),
            ),
            ("", Err(PositionError::Setup(String::new()))),
            (
                "fen invalid",
                Err(PositionError::Fen(FenError::FieldCount(1))),
            ),
            (
                "startpos moves e2e5",
                Err(PositionError::IllegalMove("e2e5".to_string())),
            ),
            (
                "fen 4k3/8/8/8/8/8/4r3/4K3 w - - 0 1 moves e1f2",
                Err(PositionError::IllegalMove("e1f2".to_string())),
            ),
        ];

        for (command, expected) in cases {
            let parsed = parse_position(&tokens(command)).map(|board| board.to_fen());
            assert_eq!(parsed, expected.map(str::to_string), "command {command:?}");
        }
    }

    #[test]
    fn parses_go_commands() {
        let cases = [
            ("", GoParams::default()),
            (
                "depth 7",
                GoParams {
                    depth: Some(7),
                    ..GoParams::default()
                },
            ),
            (
                "wtime 1000 btime 2000 winc 10 binc 20 movestogo 5",
                GoParams {
                    remaining: per_side(Some(1000), Some(2000)),
                    increment: per_side(Some(10), Some(20)),
                    moves_to_go: Some(5),
                    ..GoParams::default()
                },
            ),
            (
                "infinite",
                GoParams {
                    infinite: true,
                    ..GoParams::default()
                },
            ),
            (
                "perft 3",
                GoParams {
                    perft: Some(3),
                    ..GoParams::default()
                },
            ),
            (
                "depth deep movetime 500 ponder",
                GoParams {
                    move_time: millis(500),
                    ..GoParams::default()
                },
            ),
            (
                "wtime -20 btime 100",
                GoParams {
                    remaining: per_side(Some(0), Some(100)),
                    ..GoParams::default()
                },
            ),
        ];

        for (command, expected) in cases {
            assert_eq!(parse_go(&tokens(command)), expected, "command {command:?}");
        }
    }

    #[test]
    fn parses_hash_option() {
        let cases = [
            ("name Hash value 64", Some(64)),
            ("name hash value 1", Some(1)),
            ("name Hash value big", None),
            ("name Threads value 4", None),
            ("name Hash", None),
        ];

        for (command, expected) in cases {
            assert_eq!(
                parse_hash_option(&tokens(command)),
                expected,
                "command {command:?}"
            );
        }
    }

    #[test]
    fn derives_search_limits() {
        let cases = [
            (
                "wtime 60000 btime 30000",
                Color::White,
                None,
                millis(2000),
                millis(1000),
            ),
            (
                "wtime 60000 btime 30000",
                Color::Black,
                None,
                millis(1000),
                millis(500),
            ),
            (
                "wtime 30000 winc 1000",
                Color::White,
                None,
                millis(1750),
                millis(875),
            ),
            (
                "btime 100 movestogo 1",
                Color::Black,
                None,
                millis(50),
                millis(25),
            ),
            ("wtime -20", Color::White, None, millis(0), millis(0)),
            ("wtime 60000", Color::Black, None, None, None),
            (
                "movetime 1000 wtime 60000",
                Color::White,
                None,
                millis(950),
                None,
            ),
            ("depth 5", Color::White, Some(5), None, None),
            (
                "infinite wtime 60000 depth 9",
                Color::White,
                Some(9),
                None,
                None,
            ),
        ];

        for (command, side, depth, time, soft_time) in cases {
            let expected = SearchLimits {
                depth,
                time,
                soft_time,
            };
            let limits = parse_go(&tokens(command)).limits(side);
            assert_eq!(limits, expected, "command {command:?}, side {side:?}");
        }
    }
}
