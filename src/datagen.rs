use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::thread;
use std::time::{Duration, Instant};

use crate::board::{Board, MoveList, START_POSITION, color_flipped_fen};
use crate::cli::{default_threads, parse_argument, usage_error};
use crate::nnue::data::Outcome;
use crate::nnue::games::{self, Filter, Ply, read_header, write_header};
use crate::progress::{Ticker, bar, eta, format_duration, percent};
use crate::random::Random;
use crate::search::{SearchLimits, Searcher, SharedSearch};
use crate::strength::fresh_noise_seed;
use crate::tt::{DEFAULT_HASH_MB, TranspositionTable};
use crate::types::{Color, Move};

const DEFAULT_NODES: u64 = 2_000;
const HARD_NODE_FACTOR: u64 = 10;
const DEFAULT_DATA_DIRECTORY: &str = "data";
const RANDOM_PLIES: usize = 8;
const OPENING_SCORE_LIMIT: i32 = 500;
const WIN_ADJUDICATION_SCORE: i32 = 2_000;
const WIN_ADJUDICATION_PLIES: usize = 4;
const DRAW_ADJUDICATION_MOVE: u16 = 32;
const DRAW_ADJUDICATION_SCORE: i32 = 20;
const DRAW_ADJUDICATION_PLIES: usize = 12;
const FIFTY_MOVE_PLIES: u8 = 100;
const POLL_INTERVAL: Duration = Duration::from_millis(200);
const THREAD_SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

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
        || board.has_insufficient_material();
    drawn.then_some(Outcome::Draw)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ending {
    Rules,
    AdjudicatedWin,
    AdjudicatedDraw,
}

#[derive(Default)]
struct Adjudication {
    winning_streak: Option<(Color, usize)>,
    drawn_plies: usize,
}

impl Adjudication {
    fn observe(&mut self, move_number: u16, side: Color, score: i32) -> Option<(Outcome, Ending)> {
        if score.abs() >= WIN_ADJUDICATION_SCORE {
            let leader = if score > 0 { side } else { side.opponent() };
            let streak = match self.winning_streak {
                Some((previous, plies)) if previous == leader => plies + 1,
                _ => 1,
            };
            if streak >= WIN_ADJUDICATION_PLIES {
                return Some((Outcome::Win.for_side(leader), Ending::AdjudicatedWin));
            }
            self.winning_streak = Some((leader, streak));
        } else {
            self.winning_streak = None;
        }
        if move_number >= DRAW_ADJUDICATION_MOVE && score.abs() <= DRAW_ADJUDICATION_SCORE {
            self.drawn_plies += 1;
        } else {
            self.drawn_plies = 0;
        }
        (self.drawn_plies >= DRAW_ADJUDICATION_PLIES)
            .then_some((Outcome::Draw, Ending::AdjudicatedDraw))
    }
}

struct PlayedGame {
    game: games::Game,
    positions: u64,
    searched_plies: u64,
    searched_nodes: u64,
    ending: Ending,
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

fn play_game(random: &mut Random, shared: &Arc<SharedSearch>, nodes: u64) -> Option<PlayedGame> {
    let limits = SearchLimits {
        nodes: Some(nodes.saturating_mul(HARD_NODE_FACTOR)),
        soft_nodes: Some(nodes),
        ..SearchLimits::default()
    };
    let mut searcher = Searcher::new(limits, Arc::new(AtomicBool::new(false)), Arc::clone(shared));
    random_opening(random).and_then(|mut board| {
        let start = board.to_fen();
        let mut plies = Vec::new();
        let mut positions = 0;
        let mut searched_plies = 0;
        let mut searched_nodes = 0;
        let mut adjudication = Adjudication::default();
        let (white_outcome, ending) = loop {
            if let Some(outcome) = rules_outcome(&mut board) {
                break (outcome, Ending::Rules);
            }
            let (best_move, score) = searcher.search(&mut board, |_| {});
            searched_nodes += searcher.nodes();
            if searched_plies == 0 && score.abs() > OPENING_SCORE_LIMIT {
                return None;
            }
            let index = searched_plies;
            searched_plies += 1;
            let side = board.state.active_color;
            if let Some(adjudicated) =
                adjudication.observe(board.state.fullmove_number, side, score)
            {
                break adjudicated;
            }
            let ply = Ply {
                best_move,
                score: score as i16,
            };
            if Filter::DEFAULT.accepts(index, &board, ply) {
                positions += 1;
            }
            plies.push(ply);
            board.make_move(best_move);
        };
        Some(PlayedGame {
            game: games::Game {
                start,
                white_outcome,
                plies,
            },
            positions,
            searched_plies: searched_plies as u64,
            searched_nodes,
            ending,
        })
    })
}

#[derive(Default)]
struct Totals {
    positions: u64,
    games: u64,
    white_wins: u64,
    draws: u64,
    black_wins: u64,
    searched_plies: u64,
    searched_nodes: u64,
    adjudicated_wins: u64,
    adjudicated_draws: u64,
    discarded_openings: u64,
}

impl Totals {
    fn count(&mut self, played: &PlayedGame) {
        self.games += 1;
        self.positions += played.positions;
        self.searched_plies += played.searched_plies;
        self.searched_nodes += played.searched_nodes;
        match played.ending {
            Ending::Rules => {}
            Ending::AdjudicatedWin => self.adjudicated_wins += 1,
            Ending::AdjudicatedDraw => self.adjudicated_draws += 1,
        }
        match played.game.white_outcome {
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

    fn write_game(&mut self, played: PlayedGame) {
        if self.is_done() {
            return;
        }
        match played.game.write(&mut self.writer) {
            Ok(()) => self.totals.count(&played),
            Err(error) => self.error = Some(error),
        }
    }
}

fn generate(output: &Mutex<Output>, seed: u64, nodes: u64) {
    let mut random = Random::new(seed);
    let mut shared = Arc::new(SharedSearch::new(TranspositionTable::new(DEFAULT_HASH_MB)));
    while !output.lock().expect("output lock").is_done() {
        let game = play_game(&mut random, &shared, nodes);
        SharedSearch::exclusive_table(&mut shared).clear();
        let mut output = output.lock().expect("output lock");
        match game {
            Some(game) => output.write_game(game),
            None => output.totals.discarded_openings += 1,
        }
    }
}

fn progress_line(positions: u64, target: u64, elapsed: Duration) -> String {
    let done = positions.min(target);
    let rate = positions as f64 / elapsed.as_secs_f64().max(f64::EPSILON);
    format!(
        "{} {:5.1}%  {done}/{target}  elapsed {}  eta {}  {rate:.0}/s",
        bar(done, target),
        percent(done, target),
        format_duration(elapsed),
        eta(done, target, elapsed)
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
    file: games::Contents,
    file_bytes: u64,
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
    let by_rules = totals.games - totals.adjudicated_wins - totals.adjudicated_draws;
    eprintln!(
        "  endings            rules {by_rules} ({:.1}%), adjudicated win {} ({:.1}%), adjudicated draw {} ({:.1}%)",
        percent(by_rules, totals.games),
        totals.adjudicated_wins,
        percent(totals.adjudicated_wins, totals.games),
        totals.adjudicated_draws,
        percent(totals.adjudicated_draws, totals.games)
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
        "  settings           {} threads, {} soft nodes per move, {} hard",
        summary.threads,
        summary.nodes,
        summary.nodes.saturating_mul(HARD_NODE_FACTOR)
    );
    eprintln!(
        "  search             {:.0} nodes per move",
        ratio(totals.searched_nodes, totals.searched_plies)
    );
    eprintln!(
        "  file               {}: {} games, {} plies, {:.1} MB",
        summary.path.display(),
        summary.file.games,
        summary.file.plies,
        summary.file_bytes as f64 / 1e6
    );
}

fn default_output(nodes: u64) -> PathBuf {
    let version = env!("CARGO_PKG_VERSION");
    PathBuf::from(DEFAULT_DATA_DIRECTORY)
        .join(format!("arabica-{version}-n{nodes}.{}", games::EXTENSION))
}

fn open_games_file(path: &Path) -> io::Result<File> {
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)?;
    if file.metadata()?.len() == 0 {
        write_header(&mut file)?;
        return Ok(file);
    }
    file.seek(SeekFrom::Start(0))?;
    if !read_header(&mut file)? {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} exists and is not an arabica games file", path.display()),
        ));
    }
    Ok(file)
}

pub fn run(arguments: &[String]) -> io::Result<()> {
    let [positions, rest @ ..] = arguments else {
        return Err(usage_error(
            "arabica datagen <positions> [threads] [nodes] [output]",
        ));
    };
    let target: u64 = parse_argument(Some(positions), "positions", 0)?;
    let threads: usize = parse_argument(rest.first(), "threads", default_threads())?.max(1);
    let nodes: u64 = parse_argument(rest.get(1), "nodes", DEFAULT_NODES)?.max(1);
    let path = rest
        .get(2)
        .map_or_else(|| default_output(nodes), PathBuf::from);

    if let Some(directory) = path.parent() {
        fs::create_dir_all(directory)?;
    }
    let file = open_games_file(&path)?;
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
        let positions = || output.lock().expect("output lock").totals.positions;
        let mut ticker = Ticker::new();
        while !output.lock().expect("output lock").is_done() {
            thread::sleep(POLL_INTERVAL);
            if ticker.due() {
                ticker.show(&progress_line(positions(), target, started.elapsed()));
            }
        }
        ticker.finish(&progress_line(positions(), target, started.elapsed()));
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
        file: games::contents(&path)?,
        file_bytes: fs::metadata(&path)?.len(),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::testing::board;
    use crate::search::MATE_BOUND;
    use crate::types::Piece;

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
    fn adjudicates_decided_and_dead_drawn_games() {
        let quiet = [0; 24];
        type Verdict = Option<(usize, Outcome, Ending)>;
        let cases: [(u16, &[i32], Verdict); 10] = [
            (
                10,
                &[2_000, -2_000, 2_100, -2_200],
                Some((3, Outcome::Win, Ending::AdjudicatedWin)),
            ),
            (
                10,
                &[-3_000, 3_000, -3_000, 3_000],
                Some((3, Outcome::Loss, Ending::AdjudicatedWin)),
            ),
            (10, &[2_000, -2_000, 1_999, -2_000, 2_000, -2_000], None),
            (10, &[2_000, 2_000, 2_000, 2_000], None),
            (10, &[1_500, -1_500, 1_500, -1_500, 1_500, -1_500], None),
            (
                32,
                &quiet[..12],
                Some((11, Outcome::Draw, Ending::AdjudicatedDraw)),
            ),
            (
                32,
                &[20, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20],
                Some((11, Outcome::Draw, Ending::AdjudicatedDraw)),
            ),
            (25, &quiet[..12], None),
            (
                28,
                &quiet,
                Some((19, Outcome::Draw, Ending::AdjudicatedDraw)),
            ),
            (
                32,
                &[0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                None,
            ),
        ];

        for (first_move, scores, expected) in cases {
            let mut adjudication = Adjudication::default();
            let verdict = scores.iter().enumerate().find_map(|(step, &score)| {
                let side = if step % 2 == 0 {
                    Color::White
                } else {
                    Color::Black
                };
                let move_number = first_move + (step / 2) as u16;
                adjudication
                    .observe(move_number, side, score)
                    .map(|(outcome, ending)| (step, outcome, ending))
            });
            assert_eq!(
                verdict, expected,
                "first move {first_move}, scores {scores:?}"
            );
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
    fn default_output_names_version_and_nodes() {
        let version = env!("CARGO_PKG_VERSION");
        let cases = [
            (5_000, format!("data/arabica-{version}-n5000.games")),
            (20_000, format!("data/arabica-{version}-n20000.games")),
        ];

        for (nodes, expected) in cases {
            assert_eq!(
                default_output(nodes),
                PathBuf::from(expected),
                "nodes {nodes}"
            );
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
    fn plays_games_that_replay_into_counted_positions() {
        let seeds = [1, 2, 3, 4];

        let mut played = 0;
        for seed in seeds {
            let mut random = Random::new(seed);
            let shared = Arc::new(SharedSearch::new(TranspositionTable::new(1)));
            let game = play_game(&mut random, &shared, 200);
            let Some(played_game) = game else { continue };
            played += 1;
            let mut bytes = Vec::new();
            played_game.game.write(&mut bytes).expect("game written");
            let read = games::Game::read(&mut bytes.as_slice())
                .expect("game read")
                .expect("one game");
            assert_eq!(read, played_game.game, "seed {seed}");
            assert!(
                read.plies.len() as u64 <= played_game.searched_plies,
                "seed {seed}"
            );
            let records = read.records(&Filter::DEFAULT).expect("legal game");
            assert!(!records.is_empty(), "seed {seed}");
            assert_eq!(records.len() as u64, played_game.positions, "seed {seed}");
            for record in records {
                assert!(i32::from(record.score).abs() < MATE_BOUND, "seed {seed}");
                let kings = record
                    .pieces()
                    .filter(|&(_, piece, _)| piece == Piece::King);
                assert_eq!(kings.count(), 2, "seed {seed}");
            }
        }
        assert!(played > 0, "every opening was discarded");
    }

    #[test]
    fn appends_only_to_games_files() {
        let directory =
            std::env::temp_dir().join(format!("arabica-{}-datagen-files", std::process::id()));
        fs::create_dir_all(&directory).expect("directory created");
        let cases: [(&str, Option<&[u8]>, bool); 4] = [
            ("fresh.games", None, true),
            ("empty.games", Some(b""), true),
            ("existing.games", Some(games::MAGIC), true),
            ("positions.bin", Some(&[0; 32]), false),
        ];

        for (name, contents, accepted) in cases {
            let path = directory.join(name);
            if let Some(contents) = contents {
                fs::write(&path, contents).expect("file written");
            }
            assert_eq!(open_games_file(&path).is_ok(), accepted, "{name}");
            if accepted {
                assert_eq!(fs::read(&path).expect("file read"), games::MAGIC, "{name}");
            }
        }
        fs::remove_dir_all(&directory).expect("directory removed");
    }
}
