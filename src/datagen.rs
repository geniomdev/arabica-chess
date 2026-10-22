use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::thread;
use std::time::{Duration, Instant};

use crate::board::{Board, MoveList, START_POSITION, color_flipped_fen};
use crate::nnue::data::{Outcome, RECORD_BYTES, Record};
use crate::search::{SearchLimits, Searcher, is_mate_score};
use crate::strength::fresh_noise_seed;
use crate::tt::{DEFAULT_HASH_MB, TranspositionTable};
use crate::types::{Color, Move, Piece};

const DEFAULT_NODES: u64 = 5_000;
const DEFAULT_DATA_DIRECTORY: &str = "data";
const RANDOM_PLIES: usize = 8;
const OPENING_SCORE_LIMIT: i32 = 500;
const WIN_ADJUDICATION_SCORE: i32 = 2_500;
const WIN_ADJUDICATION_PLIES: usize = 4;
const FIFTY_MOVE_PLIES: u8 = 100;
const TERMINAL_PROGRESS_INTERVAL: Duration = Duration::from_secs(1);
const LOG_PROGRESS_INTERVAL: Duration = Duration::from_secs(60);
const PROGRESS_BAR_WIDTH: usize = 30;
const CLEAR_TO_LINE_END: &str = "\x1b[K";
const POLL_INTERVAL: Duration = Duration::from_millis(200);
const THREAD_SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

struct Random(u64);

impl Random {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

fn legal_moves(board: &mut Board) -> Vec<Move> {
    let mut moves = MoveList::new();
    board.generate_pseudo_legal(&mut moves);
    moves
        .as_slice()
        .iter()
        .copied()
        .filter(|&candidate| {
            let legal = board.make_move(candidate);
            if legal {
                board.unmake_move();
            }
            legal
        })
        .collect()
}

fn has_insufficient_material(board: &Board) -> bool {
    let heavy = board.pieces(Piece::Pawn) | board.pieces(Piece::Rook) | board.pieces(Piece::Queen);
    let minors = board.pieces(Piece::Knight) | board.pieces(Piece::Bishop);
    heavy.is_empty() && minors.count() <= 1
}

fn rules_outcome(board: &mut Board) -> Option<Outcome> {
    if legal_moves(board).is_empty() {
        let winner = board.state.active_color.opponent();
        return Some(if board.in_check() {
            Outcome::Win.for_side(winner)
        } else {
            Outcome::Draw
        });
    }
    let drawn = board.state.halfmove_clock >= FIFTY_MOVE_PLIES
        || board.is_repetition(0)
        || has_insufficient_material(board);
    drawn.then_some(Outcome::Draw)
}

fn is_quiet_sample(board: &Board, best_move: Move, score: i32) -> bool {
    !board.in_check()
        && !best_move.is_capture()
        && !best_move.is_promotion()
        && !is_mate_score(score)
}

struct Sample {
    record: Record,
    side: Color,
}

struct Game {
    samples: Vec<Sample>,
    searched_plies: u64,
    white_outcome: Outcome,
}

impl Game {
    fn into_records(self) -> impl Iterator<Item = Record> {
        let white_outcome = self.white_outcome;
        self.samples.into_iter().map(move |sample| {
            sample
                .record
                .with_outcome(white_outcome.for_side(sample.side))
        })
    }
}

fn random_opening(random: &mut Random) -> Option<Board> {
    let mut board: Board = START_POSITION.parse().expect("start position is valid");
    for _ in 0..RANDOM_PLIES {
        let moves = legal_moves(&mut board);
        if moves.is_empty() {
            return None;
        }
        board.make_move(moves[random.below(moves.len())]);
    }
    if random.below(2) == 0 {
        return Some(board);
    }
    let flipped = color_flipped_fen(&board.to_fen());
    Some(flipped.parse().expect("flipped opening is valid"))
}

fn play_game(
    random: &mut Random,
    table: TranspositionTable,
    nodes: u64,
) -> (Option<Game>, TranspositionTable) {
    let limits = SearchLimits {
        nodes: Some(nodes),
        ..SearchLimits::default()
    };
    let mut searcher = Searcher::new(limits, Arc::new(AtomicBool::new(false)), table);
    let game = random_opening(random).and_then(|mut board| {
        let mut samples = Vec::new();
        let mut searched_plies = 0;
        let mut winning_streak: Option<(Color, usize)> = None;
        let white_outcome = loop {
            if let Some(outcome) = rules_outcome(&mut board) {
                break outcome;
            }
            let (best_move, score) = searcher.search(&mut board, |_| {});
            if searched_plies == 0 && score.abs() > OPENING_SCORE_LIMIT {
                return None;
            }
            searched_plies += 1;
            let side = board.state.active_color;
            if score.abs() >= WIN_ADJUDICATION_SCORE {
                let leader = if score > 0 { side } else { side.opponent() };
                let streak = match winning_streak {
                    Some((previous, plies)) if previous == leader => plies + 1,
                    _ => 1,
                };
                if streak >= WIN_ADJUDICATION_PLIES {
                    break Outcome::Win.for_side(leader);
                }
                winning_streak = Some((leader, streak));
            } else {
                winning_streak = None;
            }
            if is_quiet_sample(&board, best_move, score) {
                samples.push(Sample {
                    record: Record::new(&board, score as i16, Outcome::Draw),
                    side,
                });
            }
            board.make_move(best_move);
        };
        Some(Game {
            samples,
            searched_plies,
            white_outcome,
        })
    });
    let (mut table, _) = searcher.into_parts();
    table.clear();
    (game, table)
}

#[derive(Default)]
struct Totals {
    positions: u64,
    games: u64,
    white_wins: u64,
    draws: u64,
    black_wins: u64,
    searched_plies: u64,
    discarded_openings: u64,
}

impl Totals {
    fn count(&mut self, game: &Game) {
        self.games += 1;
        self.searched_plies += game.searched_plies;
        match game.white_outcome {
            Outcome::Win => self.white_wins += 1,
            Outcome::Draw => self.draws += 1,
            Outcome::Loss => self.black_wins += 1,
        }
    }
}

struct Output {
    writer: BufWriter<File>,
    totals: Totals,
    target: u64,
    error: Option<io::Error>,
}

impl Output {
    fn is_done(&self) -> bool {
        self.totals.positions >= self.target || self.error.is_some()
    }

    fn write_game(&mut self, game: Game) {
        if self.is_done() {
            return;
        }
        self.totals.count(&game);
        let remaining = self.target - self.totals.positions;
        for record in game.into_records().take(remaining as usize) {
            if let Err(error) = self.writer.write_all(&record.to_bytes()) {
                self.error = Some(error);
                return;
            }
            self.totals.positions += 1;
        }
    }
}

fn generate(output: &Mutex<Output>, seed: u64, nodes: u64) {
    let mut random = Random::new(seed);
    let mut table = TranspositionTable::new(DEFAULT_HASH_MB);
    while !output.lock().expect("output lock").is_done() {
        let (game, returned) = play_game(&mut random, table, nodes);
        table = returned;
        let mut output = output.lock().expect("output lock");
        match game {
            Some(game) => output.write_game(game),
            None => output.totals.discarded_openings += 1,
        }
    }
}

fn percent(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

fn progress_line(positions: u64, target: u64, elapsed: Duration) -> String {
    let done = positions.min(target);
    let filled = (PROGRESS_BAR_WIDTH as u64 * done / target.max(1)) as usize;
    let bar = format!(
        "{}{}",
        "#".repeat(filled),
        "-".repeat(PROGRESS_BAR_WIDTH - filled)
    );
    let rate = positions as f64 / elapsed.as_secs_f64().max(f64::EPSILON);
    let eta = if positions == 0 {
        String::from("-:--:--")
    } else {
        format_duration(Duration::from_secs_f64((target - done) as f64 / rate))
    };
    format!(
        "[{bar}] {:5.1}%  {done}/{target}  elapsed {}  eta {eta}  {rate:.0}/s",
        percent(done, target),
        format_duration(elapsed)
    )
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    format!(
        "{}:{:02}:{:02}",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60
    )
}

fn ratio(part: u64, whole: u64) -> f64 {
    part as f64 / whole.max(1) as f64
}

struct Summary<'a> {
    totals: &'a Totals,
    elapsed: Duration,
    threads: usize,
    nodes: u64,
    path: &'a Path,
    file_positions: u64,
}

fn print_summary(summary: &Summary) {
    let totals = summary.totals;
    let seconds = summary.elapsed.as_secs_f64().max(f64::EPSILON);
    let rate = totals.positions as f64 / seconds;
    let attempted = totals.games + totals.discarded_openings;
    eprintln!("datagen finished");
    eprintln!("  time               {}", format_duration(summary.elapsed));
    eprintln!(
        "  positions          {} ({rate:.0}/s, {:.0}/s per thread)",
        totals.positions,
        rate / summary.threads as f64
    );
    eprintln!(
        "  games              {} ({:.2}/s), {:.1} positions and {:.1} searched plies per game",
        totals.games,
        totals.games as f64 / seconds,
        ratio(totals.positions, totals.games),
        ratio(totals.searched_plies, totals.games)
    );
    eprintln!(
        "  results            white {} ({:.1}%), draw {} ({:.1}%), black {} ({:.1}%)",
        totals.white_wins,
        percent(totals.white_wins, totals.games),
        totals.draws,
        percent(totals.draws, totals.games),
        totals.black_wins,
        percent(totals.black_wins, totals.games)
    );
    eprintln!(
        "  kept positions     {:.1}% of searched plies",
        percent(totals.positions, totals.searched_plies)
    );
    eprintln!(
        "  dropped openings   {} ({:.1}% of started games)",
        totals.discarded_openings,
        percent(totals.discarded_openings, attempted)
    );
    eprintln!(
        "  settings           {} threads, {} nodes per move",
        summary.threads, summary.nodes
    );
    eprintln!(
        "  file               {}: {} positions, {:.1} MB",
        summary.path.display(),
        summary.file_positions,
        (summary.file_positions * RECORD_BYTES as u64) as f64 / 1e6
    );
}

fn parse_argument<T: std::str::FromStr>(
    value: Option<&String>,
    name: &str,
    default: T,
) -> io::Result<T> {
    value.map_or(Ok(default), |text| {
        text.parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid {name}: {text}"),
            )
        })
    })
}

fn default_output(nodes: u64) -> PathBuf {
    let version = env!("CARGO_PKG_VERSION");
    PathBuf::from(DEFAULT_DATA_DIRECTORY).join(format!("arabica-{version}-n{nodes}.bin"))
}

pub fn run(arguments: &[String]) -> io::Result<()> {
    let [positions, rest @ ..] = arguments else {
        eprintln!("usage: arabica datagen <positions> [threads] [nodes] [output]");
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "missing arguments",
        ));
    };
    let target: u64 = parse_argument(Some(positions), "positions", 0)?;
    let default_threads = thread::available_parallelism().map_or(1, usize::from);
    let threads: usize = parse_argument(rest.first(), "threads", default_threads)?.max(1);
    let nodes: u64 = parse_argument(rest.get(1), "nodes", DEFAULT_NODES)?.max(1);
    let path = rest
        .get(2)
        .map_or_else(|| default_output(nodes), PathBuf::from);

    if let Some(directory) = path.parent() {
        fs::create_dir_all(directory)?;
    }
    let file = OpenOptions::new().create(true).append(true).open(&path)?;
    let output = Mutex::new(Output {
        writer: BufWriter::new(file),
        totals: Totals::default(),
        target,
        error: None,
    });
    eprintln!(
        "datagen: {target} positions into {}, {threads} threads, {nodes} nodes per move",
        path.display()
    );

    let started = Instant::now();
    let base_seed = fresh_noise_seed();
    thread::scope(|scope| {
        for index in 0..threads as u64 {
            let output = &output;
            let seed = base_seed.wrapping_add(index.wrapping_mul(THREAD_SEED_STRIDE));
            scope.spawn(move || generate(output, seed, nodes));
        }
        let interactive = io::stderr().is_terminal();
        let interval = if interactive {
            TERMINAL_PROGRESS_INTERVAL
        } else {
            LOG_PROGRESS_INTERVAL
        };
        let positions = || output.lock().expect("output lock").totals.positions;
        let mut last_report = Instant::now();
        while !output.lock().expect("output lock").is_done() {
            thread::sleep(POLL_INTERVAL);
            if last_report.elapsed() >= interval {
                let line = progress_line(positions(), target, started.elapsed());
                if interactive {
                    eprint!("\r{line}{CLEAR_TO_LINE_END}");
                } else {
                    eprintln!("{line}");
                }
                last_report = Instant::now();
            }
        }
        let line = progress_line(positions(), target, started.elapsed());
        if interactive {
            eprintln!("\r{line}{CLEAR_TO_LINE_END}");
        } else {
            eprintln!("{line}");
        }
    });

    let elapsed = started.elapsed();
    let mut output = output.into_inner().expect("output lock");
    if let Some(error) = output.error.take() {
        return Err(error);
    }
    output.writer.flush()?;
    print_summary(&Summary {
        totals: &output.totals,
        elapsed,
        threads,
        nodes,
        path: &path,
        file_positions: fs::metadata(&path)?.len() / RECORD_BYTES as u64,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::testing::board;
    use crate::search::MATE_BOUND;

    #[test]
    fn detects_game_end_by_rules() {
        let cases = [
            (START_POSITION, None),
            ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", Some(Outcome::Win)),
            ("k7/8/1QK5/8/8/8/8/8 b - - 0 1", Some(Outcome::Draw)),
            ("4k3/8/8/8/8/8/4P3/4K3 w - - 100 80", Some(Outcome::Draw)),
            ("4k3/8/8/8/8/8/8/4KN2 w - - 0 1", Some(Outcome::Draw)),
            ("4kb2/8/8/8/8/8/8/4KN2 w - - 0 1", None),
            ("4k3/8/8/8/8/8/8/4K2R w - - 0 1", None),
            ("6rk/5Qpp/8/8/8/8/8/6K1 b - - 0 1", None),
        ];

        for (fen, expected) in cases {
            assert_eq!(rules_outcome(&mut board(fen)), expected, "fen {fen:?}");
        }
    }

    #[test]
    fn renders_progress_lines() {
        let cases = [
            (
                0,
                100,
                0,
                "[------------------------------]   0.0%  0/100  elapsed 0:00:00  eta -:--:--  0/s",
            ),
            (
                25_000_000,
                100_000_000,
                10_000,
                "[#######-----------------------]  25.0%  25000000/100000000  elapsed 2:46:40  eta 8:20:00  2500/s",
            ),
            (
                100,
                100,
                50,
                "[##############################] 100.0%  100/100  elapsed 0:00:50  eta 0:00:00  2/s",
            ),
        ];

        for (positions, target, seconds, expected) in cases {
            assert_eq!(
                progress_line(positions, target, Duration::from_secs(seconds)),
                expected,
                "{positions}/{target} after {seconds} s"
            );
        }
    }

    #[test]
    fn formats_durations() {
        let cases = [
            (0, "0:00:00"),
            (59, "0:00:59"),
            (61, "0:01:01"),
            (3_600, "1:00:00"),
            (45_296, "12:34:56"),
            (100 * 3_600, "100:00:00"),
        ];

        for (seconds, expected) in cases {
            assert_eq!(
                format_duration(Duration::from_secs(seconds)),
                expected,
                "{seconds} s"
            );
        }
    }

    #[test]
    fn default_output_names_version_and_nodes() {
        let version = env!("CARGO_PKG_VERSION");
        let cases = [
            (5_000, format!("data/arabica-{version}-n5000.bin")),
            (20_000, format!("data/arabica-{version}-n20000.bin")),
        ];

        for (nodes, expected) in cases {
            assert_eq!(default_output(nodes), PathBuf::from(expected), "nodes {nodes}");
        }
    }

    #[test]
    fn random_openings_hand_the_move_to_both_sides() {
        let seeds = 1..=64;

        let mut to_move = [0; 2];
        for seed in seeds {
            let mut random = Random::new(seed);
            let Some(opening) = random_opening(&mut random) else {
                continue;
            };
            assert_eq!(
                usize::from(opening.state.fullmove_number),
                RANDOM_PLIES / 2 + 1,
                "seed {seed}"
            );
            to_move[opening.state.active_color.index()] += 1;
        }
        assert!(
            to_move.iter().all(|&count| count > 0),
            "side to move {to_move:?}"
        );
    }

    #[test]
    fn plays_games_into_valid_samples() {
        let seeds = [1, 2, 3, 4];

        let mut played = 0;
        for seed in seeds {
            let mut random = Random::new(seed);
            let (game, _) = play_game(&mut random, TranspositionTable::new(1), 200);
            let Some(game) = game else { continue };
            played += 1;
            assert!(!game.samples.is_empty(), "seed {seed}");
            let outcome = game.white_outcome;
            let sides: Vec<Color> = game.samples.iter().map(|sample| sample.side).collect();
            for (record, side) in game.into_records().zip(sides) {
                assert_eq!(record.outcome, outcome.for_side(side), "seed {seed}");
                assert!(i32::from(record.score).abs() < MATE_BOUND, "seed {seed}");
                let kings = record
                    .pieces()
                    .filter(|&(_, piece, _)| piece == Piece::King);
                assert_eq!(kings.count(), 2, "seed {seed}");
            }
        }
        assert!(played > 0, "every opening was discarded");
    }
}
