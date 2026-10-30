use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use crate::board::{Board, START_POSITION, color_flipped_fen};
use crate::cli::{default_threads, invalid_data, parse_argument, usage_error};
use crate::nnue::data::Outcome;
use crate::nnue::games::{self, Filter, Ply, read_header, write_header};
use crate::progress::{Ticker, bar, eta, format_duration, per_second, percent, ratio};
use crate::random::Random;
use crate::search::{SearchLimits, Searcher, SharedSearch};
use crate::strength::fresh_noise_seed;
use crate::syzygy::{Tablebases, Wdl};
use crate::tt::DEFAULT_HASH_MB;
use crate::types::Color;

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
const TABLEBASE_CONTINUATION_PLIES: usize = 16;
const POLL_INTERVAL: Duration = Duration::from_millis(200);
const THREAD_SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

fn rules_outcome(board: &mut Board) -> Option<Outcome> {
    if board.legal_moves().is_empty() {
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
    Tablebase,
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

fn tablebase_outcome(tablebases: &Tablebases, board: &mut Board) -> Option<Outcome> {
    if board.state.halfmove_clock != 0 {
        return None;
    }
    let side = board.state.active_color;
    Some(match tablebases.probe_wdl(board)? {
        Wdl::Win => Outcome::Win.for_side(side),
        Wdl::Loss => Outcome::Loss.for_side(side),
        Wdl::CursedWin | Wdl::Draw | Wdl::BlessedLoss => Outcome::Draw,
    })
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
        let moves = board.legal_moves();
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

struct GameSetup<'a> {
    shared: &'a Arc<SharedSearch>,
    nodes: u64,
    tablebases: Option<&'a Arc<Tablebases>>,
}

fn play_game(random: &mut Random, setup: &GameSetup) -> Option<PlayedGame> {
    play_from(random_opening(random)?, setup, true)
}

fn play_from(mut board: Board, setup: &GameSetup, balanced_start: bool) -> Option<PlayedGame> {
    let nodes = setup.nodes;
    let limits = SearchLimits {
        nodes: Some(nodes.saturating_mul(HARD_NODE_FACTOR)),
        soft_nodes: Some(nodes),
        ..SearchLimits::default()
    };
    let mut searcher = Searcher::new(limits, Arc::default(), Arc::clone(setup.shared));
    if let Some(tablebases) = setup.tablebases {
        searcher = searcher.with_root_tablebases(Arc::clone(tablebases));
    }
    let start = board.to_fen();
    let mut plies = Vec::new();
    let mut positions = 0;
    let mut searched_plies = 0;
    let mut searched_nodes = 0;
    let mut adjudication = Adjudication::default();
    let mut settled: Option<(Outcome, usize)> = None;
    let (white_outcome, ending) = loop {
        if let Some(outcome) = rules_outcome(&mut board) {
            break match settled {
                Some((tablebase_outcome, _)) => (tablebase_outcome, Ending::Tablebase),
                None => (outcome, Ending::Rules),
            };
        }
        if settled.is_none() {
            settled = setup
                .tablebases
                .and_then(|tablebases| tablebase_outcome(tablebases, &mut board))
                .map(|outcome| (outcome, TABLEBASE_CONTINUATION_PLIES));
        }
        if let Some((outcome, 0)) = settled {
            break (outcome, Ending::Tablebase);
        }
        let (best_move, score) = searcher.search(&mut board, |_| {});
        searched_nodes += searcher.nodes();
        if balanced_start && searched_plies == 0 && score.abs() > OPENING_SCORE_LIMIT {
            return None;
        }
        let index = searched_plies;
        searched_plies += 1;
        let side = board.state.active_color;
        if settled.is_none()
            && let Some(adjudicated) =
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
        if let Some((_, remaining)) = &mut settled {
            *remaining -= 1;
        }
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
    tablebase_endings: u64,
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
            Ending::Tablebase => self.tablebase_endings += 1,
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

fn generate(output: &Mutex<Output>, seed: u64, nodes: u64, tablebases: Option<&Arc<Tablebases>>) {
    let mut random = Random::new(seed);
    let mut shared = Arc::new(SharedSearch::new(DEFAULT_HASH_MB));
    while !output.lock().expect("output lock").is_done() {
        let setup = GameSetup {
            shared: &shared,
            nodes,
            tablebases,
        };
        let game = play_game(&mut random, &setup);
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
    let rate = per_second(positions, elapsed);
    format!(
        "{} {:5.1}%  {done}/{target}  elapsed {}  eta {}  {rate:.0}/s",
        bar(done, target),
        percent(done, target),
        format_duration(elapsed),
        eta(done, target, elapsed)
    )
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
    let rate = per_second(totals.positions, summary.elapsed);
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
        per_second(totals.games, summary.elapsed),
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
    let by_rules = totals.games
        - totals.adjudicated_wins
        - totals.adjudicated_draws
        - totals.tablebase_endings;
    eprintln!(
        "  endings            rules {by_rules} ({:.1}%), adjudicated win {} ({:.1}%), adjudicated draw {} ({:.1}%), tablebase {} ({:.1}%)",
        percent(by_rules, totals.games),
        totals.adjudicated_wins,
        percent(totals.adjudicated_wins, totals.games),
        totals.adjudicated_draws,
        percent(totals.adjudicated_draws, totals.games),
        totals.tablebase_endings,
        percent(totals.tablebase_endings, totals.games)
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
        return Err(invalid_data(format!(
            "{} exists and is not an arabica games file",
            path.display()
        )));
    }
    Ok(file)
}

fn open_tablebases(paths: &str) -> io::Result<Arc<Tablebases>> {
    let tablebases = Tablebases::open(paths)?;
    if tablebases.is_empty() {
        return Err(invalid_data(format!("no Syzygy tables in {paths}")));
    }
    Ok(Arc::new(tablebases))
}

pub fn run(arguments: &[String]) -> io::Result<()> {
    let [positions, rest @ ..] = arguments else {
        return Err(usage_error(
            "arabica datagen <positions> [threads] [nodes] [output] [syzygy]",
        ));
    };
    let target: u64 = parse_argument(Some(positions), "positions", 0)?;
    let threads: usize = parse_argument(rest.first(), "threads", default_threads())?.max(1);
    let nodes: u64 = parse_argument(rest.get(1), "nodes", DEFAULT_NODES)?.max(1);
    let path = rest
        .get(2)
        .map_or_else(|| default_output(nodes), PathBuf::from);
    let tablebases = rest
        .get(3)
        .map(|paths| open_tablebases(paths))
        .transpose()?;

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
    if let Some(tablebases) = &tablebases {
        eprintln!(
            "datagen: {} tablebases up to {} pieces settle results, {TABLEBASE_CONTINUATION_PLIES} plies are played on after",
            tablebases.len(),
            tablebases.max_pieces()
        );
    }

    let started = Instant::now();
    let base_seed = fresh_noise_seed();
    thread::scope(|scope| {
        for index in 0..threads as u64 {
            let output = &output;
            let tablebases = tablebases.as_ref();
            let seed = base_seed.wrapping_add(index.wrapping_mul(THREAD_SEED_STRIDE));
            scope.spawn(move || generate(output, seed, nodes, tablebases));
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
    use crate::board::testing::{board, scratch_path};
    use crate::search::MATE_BOUND;
    use crate::syzygy::testing::local_tablebases;
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
            let shared = Arc::new(SharedSearch::new(1));
            let setup = GameSetup {
                shared: &shared,
                nodes: 200,
                tablebases: None,
            };
            let game = play_game(&mut random, &setup);
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
    fn settles_results_with_tablebases() {
        let Some(tablebases) = local_tablebases() else {
            return;
        };
        let tablebases = Arc::new(tablebases);
        let kbn_vs_k = "8/8/4k3/8/8/8/1B6/1K2N3 w - - 0 1";
        let cases = [
            (kbn_vs_k, true, Some(Outcome::Win), true),
            (
                "8/k7/8/R7/8/8/4K3/8 b - - 0 1",
                true,
                Some(Outcome::Win),
                true,
            ),
            (
                "8/8/8/3k4/2n5/8/7B/4KB2 w - - 0 1",
                true,
                Some(Outcome::Draw),
                true,
            ),
            (
                "2b2N1k/8/8/8/1K6/8/1b6/8 w - - 0 1",
                true,
                Some(Outcome::Draw),
                true,
            ),
            (
                "6k1/7p/8/8/8/2K5/3R3r/1R6 w - - 0 1",
                true,
                Some(Outcome::Win),
                false,
            ),
            ("8/8/4k3/8/8/8/1B6/1K2N3 w - - 1 1", true, None, false),
            (kbn_vs_k, false, None, false),
        ];

        for (fen, with_tablebases, settled_outcome, settled_at_start) in cases {
            let shared = Arc::new(SharedSearch::new(1));
            let setup = GameSetup {
                shared: &shared,
                nodes: 200,
                tablebases: with_tablebases.then_some(&tablebases),
            };
            let played = play_from(board(fen), &setup, false).expect("game played");
            let plies = played.game.plies.len();
            match settled_outcome {
                Some(outcome) => {
                    assert_eq!(played.ending, Ending::Tablebase, "fen {fen:?}");
                    assert_eq!(played.game.white_outcome, outcome, "fen {fen:?}");
                }
                None => assert_ne!(played.ending, Ending::Tablebase, "fen {fen:?}"),
            }
            if settled_at_start {
                assert!(
                    plies <= TABLEBASE_CONTINUATION_PLIES,
                    "fen {fen:?}, {plies} plies"
                );
            }
            let records = played.game.records(&Filter::DEFAULT).expect("legal game");
            assert_eq!(records.len() as u64, played.positions, "fen {fen:?}");
        }
    }

    #[test]
    fn appends_only_to_games_files() {
        let directory = scratch_path("datagen-files");
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

    fn played(positions: u64, white_outcome: Outcome, ending: Ending) -> PlayedGame {
        PlayedGame {
            game: games::Game {
                start: START_POSITION.to_owned(),
                white_outcome,
                plies: Vec::new(),
            },
            positions,
            searched_plies: positions + 1,
            searched_nodes: 100 * positions,
            ending,
        }
    }

    #[test]
    fn output_counts_written_games_up_to_the_target() {
        let directory = scratch_path("datagen-output");
        fs::create_dir_all(&directory).expect("directory created");
        let mixed = [
            (4, Outcome::Win, Ending::Rules),
            (3, Outcome::Draw, Ending::AdjudicatedDraw),
            (5, Outcome::Loss, Ending::AdjudicatedWin),
            (2, Outcome::Win, Ending::AdjudicatedWin),
            (3, Outcome::Draw, Ending::Tablebase),
        ];
        let cases: [(&str, u64, bool, [u64; 10]); 4] = [
            ("all", 100, false, [5, 17, 2, 2, 1, 2, 1, 1, 22, 1_700]),
            ("cut", 7, false, [2, 7, 1, 1, 0, 0, 1, 0, 9, 700]),
            ("zero", 0, false, [0; 10]),
            ("failed", 100, true, [0; 10]),
        ];

        for (name, target, failed, expected) in cases {
            let path = directory.join(format!("{name}.games"));
            let mut output = Output {
                writer: BufWriter::new(open_games_file(&path).expect("file opened")),
                totals: Totals::default(),
                target,
                error: failed.then(|| io::Error::other("disk full")),
            };
            for (positions, outcome, ending) in mixed {
                output.write_game(played(positions, outcome, ending));
            }
            output.writer.flush().expect("file flushed");
            let totals = &output.totals;
            let counted = [
                totals.games,
                totals.positions,
                totals.white_wins,
                totals.draws,
                totals.black_wins,
                totals.adjudicated_wins,
                totals.adjudicated_draws,
                totals.tablebase_endings,
                totals.searched_plies,
                totals.searched_nodes,
            ];
            assert_eq!(counted, expected, "{name}");
            let file = games::contents(&path).expect("file read");
            assert_eq!(file.games, totals.games, "{name}");
        }
        fs::remove_dir_all(&directory).expect("directory removed");
    }
}
