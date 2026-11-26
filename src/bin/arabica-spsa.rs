use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

use arabica::cli::{default_threads, invalid_data, parse_argument, usage_error};
use arabica::progress::{Ticker, bar, eta, format_duration};
use arabica::random::{Random, splitmix64};

const USAGE: &str = "arabica-spsa <tune engine> <state file> [--iterations PAIRS] \
[--batch PAIRS] [--tc 8+0.08] [--hash 16] [--concurrency N] [--book FILE] \
[--fastchess FILE] [--seed N] [--log-every PAIRS]";
const DEFAULT_ITERATIONS: u64 = 20_000;
const DEFAULT_BATCH: u64 = 4;
const DEFAULT_TC: &str = "8+0.08";
const DEFAULT_HASH: u32 = 16;
const DEFAULT_BOOK: &str = "data/UHO_Lichess_4852_v1.epd";
const DEFAULT_FASTCHESS: &str = "bin/fastchess";
const DEFAULT_SEED: u64 = 0x5EED_5B5A;
const DEFAULT_LOG_EVERY: u64 = 200;
const TRAJECTORY_EXTENSION: &str = "trajectory.csv";
const STABILITY_FRACTION: f64 = 0.1;
const LEARNING_RATE_DECAY: f64 = 0.602;
const PERTURBATION_DECAY: f64 = 0.101;
const ITERATIONS_HEADER: &str = "iterations";

#[derive(Clone, Debug, PartialEq)]
struct Parameter {
    name: String,
    value: f64,
    default: i32,
    min: i32,
    max: i32,
    c_end: f64,
    r_end: f64,
}

impl Parameter {
    fn parse(line: &str) -> Option<Self> {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let [name, "int", value, min, max, c_end, r_end] = fields[..] else {
            return None;
        };
        let default: i32 = value.parse().ok()?;
        Some(Self {
            name: name.to_string(),
            value: f64::from(default),
            default,
            min: min.parse().ok()?,
            max: max.parse().ok()?,
            c_end: c_end.parse().ok()?,
            r_end: r_end.parse().ok()?,
        })
    }

    fn clamp(&self, value: f64) -> f64 {
        value.clamp(f64::from(self.min), f64::from(self.max))
    }

    fn rounded(&self, value: f64) -> i32 {
        self.clamp(value).round() as i32
    }
}

struct Schedule {
    iterations: f64,
    stability: f64,
}

impl Schedule {
    fn new(iterations: u64) -> Self {
        let iterations = iterations as f64;
        Self {
            iterations,
            stability: STABILITY_FRACTION * iterations,
        }
    }

    fn perturbation(&self, parameter: &Parameter, iteration: u64) -> f64 {
        let start = parameter.c_end * self.iterations.powf(PERTURBATION_DECAY);
        start / (iteration as f64 + 1.0).powf(PERTURBATION_DECAY)
    }

    fn learning_rate(&self, parameter: &Parameter, iteration: u64) -> f64 {
        let final_rate = parameter.r_end * parameter.c_end * parameter.c_end;
        let start = final_rate * (self.stability + self.iterations).powf(LEARNING_RATE_DECAY);
        start / (self.stability + iteration as f64 + 1.0).powf(LEARNING_RATE_DECAY)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Outcome {
    wins: u64,
    losses: u64,
    draws: u64,
}

impl Outcome {
    fn parse(output: &str) -> Option<Self> {
        let line = output
            .lines()
            .rev()
            .find(|line| line.starts_with("Games:"))?;
        let mut counts = HashMap::new();
        for field in line.split(',') {
            let (name, value) = field.split_once(':')?;
            counts.insert(name.trim(), value.trim());
        }
        let count = |name: &str| counts.get(name)?.parse().ok();
        Some(Self {
            wins: count("Wins")?,
            losses: count("Losses")?,
            draws: count("Draws")?,
        })
    }

    fn balance(self) -> f64 {
        self.wins as f64 - self.losses as f64
    }

    fn games(self) -> u64 {
        self.wins + self.losses + self.draws
    }

    fn add(&mut self, other: Self) {
        self.wins += other.wins;
        self.losses += other.losses;
        self.draws += other.draws;
    }
}

struct Task {
    iteration: u64,
    seed: u64,
    flips: Vec<f64>,
    plus: Vec<i32>,
    minus: Vec<i32>,
}

struct Tuning {
    parameters: Vec<Parameter>,
    dispatched: u64,
    finished: u64,
    outcome: Outcome,
    failure: Option<String>,
}

impl Tuning {
    fn next_task(&mut self, settings: &Settings, schedule: &Schedule) -> Option<Task> {
        if self.failure.is_some() || self.dispatched >= settings.iterations {
            return None;
        }
        let iteration = self.dispatched;
        self.dispatched += settings.batch;
        let seed = splitmix64(settings.seed ^ iteration);
        let mut random = Random::new(seed);
        let flips: Vec<f64> = self
            .parameters
            .iter()
            .map(|_| {
                if random.next_u64() & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let shifted = |sign: f64| -> Vec<i32> {
            self.parameters
                .iter()
                .zip(&flips)
                .map(|(parameter, flip)| {
                    let shift = schedule.perturbation(parameter, iteration) * flip * sign;
                    parameter.rounded(parameter.value + shift)
                })
                .collect()
        };
        Some(Task {
            iteration,
            seed,
            plus: shifted(1.0),
            minus: shifted(-1.0),
            flips,
        })
    }

    fn apply(&mut self, task: &Task, outcome: Outcome, settings: &Settings, schedule: &Schedule) {
        for (parameter, flip) in self.parameters.iter_mut().zip(&task.flips) {
            let step = schedule.learning_rate(parameter, task.iteration) * outcome.balance()
                / (schedule.perturbation(parameter, task.iteration) * flip);
            parameter.value = parameter.clamp(parameter.value + step);
        }
        self.finished += settings.batch;
        self.outcome.add(outcome);
    }

    fn render(&self) -> String {
        let mut text = format!("{ITERATIONS_HEADER} {}\n", self.finished);
        for parameter in &self.parameters {
            text.push_str(&format!("{} {:.4}\n", parameter.name, parameter.value));
        }
        text
    }

    fn trajectory_header(&self) -> String {
        let names: Vec<&str> = self
            .parameters
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect();
        format!("pairs,{}\n", names.join(","))
    }

    fn trajectory_row(&self) -> String {
        let values: Vec<String> = self
            .parameters
            .iter()
            .map(|parameter| format!("{:.2}", parameter.value))
            .collect();
        format!("{},{}\n", self.finished, values.join(","))
    }

    fn reached_log_point(&self, batch: u64, every: u64) -> bool {
        self.finished / every > self.finished.saturating_sub(batch) / every
    }

    fn restore(&mut self, saved: &str) -> io::Result<()> {
        let values: HashMap<&str, &str> = saved
            .lines()
            .filter_map(|line| line.split_once(' '))
            .collect();
        if let Some(finished) = values.get(ITERATIONS_HEADER) {
            self.finished = finished
                .parse()
                .map_err(|_| invalid_data(format!("bad iteration count: {finished}")))?;
            self.dispatched = self.finished;
        }
        for parameter in &mut self.parameters {
            if let Some(value) = values.get(parameter.name.as_str()) {
                let value: f64 = value
                    .parse()
                    .map_err(|_| invalid_data(format!("bad value for {}", parameter.name)))?;
                parameter.value = parameter.clamp(value);
            }
        }
        Ok(())
    }
}

struct Settings {
    engine: PathBuf,
    state: PathBuf,
    iterations: u64,
    batch: u64,
    tc: String,
    hash: u32,
    concurrency: usize,
    book: PathBuf,
    fastchess: PathBuf,
    seed: u64,
    log_every: u64,
}

impl Settings {
    fn parse(arguments: &[String]) -> io::Result<Self> {
        let [engine, state, options @ ..] = arguments else {
            return Err(usage_error(USAGE));
        };
        let mut flags = HashMap::new();
        let mut rest = options.iter();
        while let Some(flag) = rest.next() {
            let name = flag.strip_prefix("--").ok_or_else(|| usage_error(USAGE))?;
            let value = rest.next().ok_or_else(|| usage_error(USAGE))?;
            flags.insert(name, value);
        }
        let flag = |name: &str| flags.get(name).copied();
        let path = |name: &str, default: &str| {
            fs::canonicalize(flag(name).map_or(default, String::as_str))
        };
        Ok(Self {
            engine: fs::canonicalize(engine)?,
            state: PathBuf::from(state),
            iterations: parse_argument(flag("iterations"), "iterations", DEFAULT_ITERATIONS)?,
            batch: parse_argument(flag("batch"), "batch", DEFAULT_BATCH)?.max(1),
            tc: flag("tc").map_or(DEFAULT_TC.to_string(), String::clone),
            hash: parse_argument(flag("hash"), "hash", DEFAULT_HASH)?,
            concurrency: parse_argument(
                flag("concurrency"),
                "concurrency",
                default_threads().saturating_sub(1).max(1),
            )?,
            book: path("book", DEFAULT_BOOK)?,
            fastchess: path("fastchess", DEFAULT_FASTCHESS)?,
            seed: parse_argument(flag("seed"), "seed", DEFAULT_SEED)?,
            log_every: parse_argument(flag("log-every"), "log every", DEFAULT_LOG_EVERY)?.max(1),
        })
    }
}

fn main() -> io::Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let settings = Settings::parse(&arguments)?;
    let work = std::env::temp_dir().join(format!("arabica-spsa-{}", std::process::id()));
    fs::create_dir_all(&work)?;
    let engine = work.join("arabica-tune");
    fs::copy(&settings.engine, &engine)?;
    let parameters = read_parameters(&engine)?;
    let mut tuning = Tuning {
        parameters,
        dispatched: 0,
        finished: 0,
        outcome: Outcome::default(),
        failure: None,
    };
    if let Ok(saved) = fs::read_to_string(&settings.state) {
        tuning.restore(&saved)?;
        eprintln!(
            "spsa: resuming {} at {} pairs",
            settings.state.display(),
            tuning.finished
        );
    }
    eprintln!(
        "spsa: {} parameters, {} pairs in batches of {}, tc {}, concurrency {}",
        tuning.parameters.len(),
        settings.iterations,
        settings.batch,
        settings.tc,
        settings.concurrency
    );
    start_trajectory(&settings, &tuning)?;
    let result = run(&settings, &engine, &work, tuning);
    fs::remove_dir_all(&work)?;
    let tuning = result?;
    if tuning.finished % settings.log_every != 0 {
        append_trajectory(&settings, &tuning)?;
    }
    print_summary(&tuning.parameters);
    Ok(())
}

fn read_parameters(engine: &Path) -> io::Result<Vec<Parameter>> {
    let mut child = Command::new(engine)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(b"tunables\nquit\n")?;
    let output = child.wait_with_output()?;
    let parameters: Vec<Parameter> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(Parameter::parse)
        .collect();
    if parameters.is_empty() {
        return Err(invalid_data(format!(
            "{} lists no tunables; build it with --features tune",
            engine.display()
        )));
    }
    Ok(parameters)
}

fn run(settings: &Settings, engine: &Path, work: &Path, tuning: Tuning) -> io::Result<Tuning> {
    let schedule = Schedule::new(settings.iterations);
    let started = Instant::now();
    let resumed_at = tuning.finished;
    let shared = Mutex::new((tuning, Ticker::new()));
    thread::scope(|scope| {
        for worker in 0..settings.concurrency {
            let worker_dir = work.join(format!("worker-{worker}"));
            let (shared, schedule) = (&shared, &schedule);
            scope.spawn(move || {
                fs::create_dir_all(&worker_dir).expect("worker directory is creatable");
                loop {
                    let task = {
                        let (tuning, _) = &mut *shared.lock().expect("no worker panics");
                        tuning.next_task(settings, schedule)
                    };
                    let Some(task) = task else {
                        break;
                    };
                    let names = {
                        let (tuning, _) = &*shared.lock().expect("no worker panics");
                        tuning
                            .parameters
                            .iter()
                            .map(|parameter| parameter.name.clone())
                            .collect::<Vec<_>>()
                    };
                    let played = play(settings, engine, &worker_dir, &names, &task);
                    let (tuning, ticker) = &mut *shared.lock().expect("no worker panics");
                    match played {
                        Ok(outcome) => {
                            tuning.apply(&task, outcome, settings, schedule);
                            if let Err(error) =
                                save(&settings.state, &tuning.render()).and_then(|()| {
                                    if tuning.reached_log_point(settings.batch, settings.log_every)
                                    {
                                        append_trajectory(settings, tuning)
                                    } else {
                                        Ok(())
                                    }
                                })
                            {
                                tuning.failure = Some(format!("saving state: {error}"));
                            }
                            let line = progress(tuning, settings, resumed_at, started);
                            if ticker.due() {
                                ticker.show(&line);
                            }
                        }
                        Err(error) => tuning.failure = Some(error.to_string()),
                    }
                }
            });
        }
    });
    let (tuning, ticker) = shared.into_inner().expect("no worker panics");
    ticker.finish(&progress(&tuning, settings, resumed_at, started));
    match tuning.failure {
        Some(failure) => Err(io::Error::other(failure)),
        None => Ok(tuning),
    }
}

fn play(
    settings: &Settings,
    engine: &Path,
    worker_dir: &Path,
    names: &[String],
    task: &Task,
) -> io::Result<Outcome> {
    let mut command = Command::new(&settings.fastchess);
    command.current_dir(worker_dir);
    for (side, values) in [("plus", &task.plus), ("minus", &task.minus)] {
        command
            .arg("-engine")
            .arg(format!("cmd={}", engine.display()))
            .arg(format!("name={side}"));
        for (name, value) in names.iter().zip(values) {
            command.arg(format!("option.{name}={value}"));
        }
    }
    command
        .args(["-each", &format!("tc={}", settings.tc)])
        .arg(format!("option.Hash={}", settings.hash))
        .args(["-openings", &format!("file={}", settings.book.display())])
        .args([
            "format=epd",
            "order=random",
            "-srand",
            &task.seed.to_string(),
        ])
        .args([
            "-rounds",
            &settings.batch.to_string(),
            "-games",
            "2",
            "-repeat",
        ])
        .args(["-concurrency", "1", "-recover", "-config"])
        .arg(format!(
            "outname={}",
            worker_dir.join("config.json").display()
        ));
    let output = command
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    Outcome::parse(&stdout)
        .filter(|outcome| outcome.games() > 0)
        .ok_or_else(|| {
            invalid_data(format!(
                "fastchess produced no result (exit {})",
                output.status
            ))
        })
}

fn trajectory_path(settings: &Settings) -> PathBuf {
    settings.state.with_extension(TRAJECTORY_EXTENSION)
}

fn start_trajectory(settings: &Settings, tuning: &Tuning) -> io::Result<()> {
    let path = trajectory_path(settings);
    if path.exists() {
        return Ok(());
    }
    fs::write(path, tuning.trajectory_header() + &tuning.trajectory_row())
}

fn append_trajectory(settings: &Settings, tuning: &Tuning) -> io::Result<()> {
    fs::OpenOptions::new()
        .append(true)
        .open(trajectory_path(settings))?
        .write_all(tuning.trajectory_row().as_bytes())
}

fn save(path: &Path, contents: &str) -> io::Result<()> {
    let staging = path.with_extension("tmp");
    fs::write(&staging, contents)?;
    fs::rename(staging, path)
}

fn progress(tuning: &Tuning, settings: &Settings, resumed_at: u64, started: Instant) -> String {
    let outcome = tuning.outcome;
    let elapsed = started.elapsed();
    let done = tuning.finished.saturating_sub(resumed_at);
    let left = settings.iterations.saturating_sub(resumed_at);
    format!(
        "{} {}/{} pairs, {} games (+{} -{} ={}), {} elapsed, eta {}",
        bar(tuning.finished, settings.iterations),
        tuning.finished,
        settings.iterations,
        outcome.games(),
        outcome.wins,
        outcome.losses,
        outcome.draws,
        format_duration(elapsed),
        eta(done, left, elapsed)
    )
}

fn print_summary(parameters: &[Parameter]) {
    let width = parameters
        .iter()
        .map(|parameter| parameter.name.len())
        .max()
        .unwrap_or(0);
    println!("{:width$}  default    tuned   change", "parameter");
    for parameter in parameters {
        let tuned = parameter.rounded(parameter.value);
        println!(
            "{:width$}  {:>7}  {:>7}  {:>+7}",
            parameter.name,
            parameter.default,
            tuned,
            tuned - parameter.default
        );
    }
    println!("\nsrc/params.rs:");
    for parameter in parameters {
        println!(
            "    {} = {}, {}, {}, {:?};",
            parameter.name,
            parameter.rounded(parameter.value),
            parameter.min,
            parameter.max,
            parameter.c_end
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameter(c_end: f64) -> Parameter {
        Parameter {
            name: String::from("margin"),
            value: 100.0,
            default: 100,
            min: 0,
            max: 200,
            c_end,
            r_end: 0.002,
        }
    }

    #[test]
    fn parses_tunable_lines() {
        let cases = [
            ("margin, int, 100, 0, 200, 8, 0.002", Some(parameter(8.0))),
            ("margin,int,100,0,200,8,0.002", Some(parameter(8.0))),
            ("margin, float, 100, 0, 200, 8, 0.002", None),
            ("margin, int, 100, 0, 200, 8", None),
            ("margin, int, wide, 0, 200, 8, 0.002", None),
            ("Arabica 0.4.11", None),
        ];

        for (line, expected) in cases {
            assert_eq!(Parameter::parse(line), expected, "line {line:?}");
        }
    }

    #[test]
    fn decays_gains_to_their_final_values() {
        let iterations = 10_000;
        let schedule = Schedule::new(iterations);
        let cases = [
            (8.0, iterations - 1, 8.0, 0.002 * 8.0 * 8.0),
            (0.5, iterations - 1, 0.5, 0.002 * 0.5 * 0.5),
            (8.0, 0, 8.0 * 10_000_f64.powf(0.101), 0.0),
        ];

        for (c_end, iteration, perturbation, final_rate) in cases {
            let parameter = parameter(c_end);
            let shown = schedule.perturbation(&parameter, iteration);
            assert!(
                (shown - perturbation).abs() < 1e-9,
                "c_end {c_end}, iteration {iteration}: perturbation {shown}"
            );
            let rate = schedule.learning_rate(&parameter, iteration);
            if final_rate > 0.0 {
                assert!(
                    (rate - final_rate).abs() < 1e-9,
                    "c_end {c_end}, iteration {iteration}: rate {rate}"
                );
            } else {
                assert!(
                    rate > 0.002 * c_end * c_end,
                    "early rate {rate} is not larger"
                );
            }
        }
    }

    #[test]
    fn parses_fastchess_outcomes() {
        let cases = [
            (
                "Games: 8, Wins: 3, Losses: 2, Draws: 3, Points: 4.5 (56.25 %)",
                Some((3, 2, 3)),
            ),
            (
                "Games: 2, Wins: 0, Losses: 0, Draws: 2, Points: 1.0 (50.00 %)\n\
                 Games: 8, Wins: 1, Losses: 4, Draws: 3, Points: 2.5 (31.25 %)",
                Some((1, 4, 3)),
            ),
            ("Finished game 1", None),
            ("Games: 8, Wins: many, Losses: 2, Draws: 3", None),
        ];

        for (output, expected) in cases {
            let parsed =
                Outcome::parse(output).map(|outcome| (outcome.wins, outcome.losses, outcome.draws));
            assert_eq!(parsed, expected, "output {output:?}");
        }
    }

    #[test]
    fn moves_parameters_toward_the_winning_side() {
        let settings = Settings {
            engine: PathBuf::new(),
            state: PathBuf::new(),
            iterations: 100,
            batch: 1,
            tc: String::new(),
            hash: 1,
            concurrency: 1,
            book: PathBuf::new(),
            fastchess: PathBuf::new(),
            seed: 1,
            log_every: 200,
        };
        let schedule = Schedule::new(settings.iterations);
        let cases = [
            ((2, 0), 1.0, 1),
            ((2, 0), -1.0, -1),
            ((0, 2), 1.0, -1),
            ((1, 1), 1.0, 0),
        ];

        for ((wins, losses), flip, direction) in cases {
            let mut tuning = Tuning {
                parameters: vec![parameter(8.0)],
                dispatched: 1,
                finished: 0,
                outcome: Outcome::default(),
                failure: None,
            };
            let task = Task {
                iteration: 0,
                seed: 0,
                flips: vec![flip],
                plus: vec![0],
                minus: vec![0],
            };
            let outcome = Outcome {
                wins,
                losses,
                draws: 0,
            };
            tuning.apply(&task, outcome, &settings, &schedule);
            let moved = tuning.parameters[0].value - 100.0;
            assert_eq!(
                moved.partial_cmp(&0.0).map(|order| order as i32),
                Some(direction),
                "wins {wins}, losses {losses}, flip {flip}: moved {moved}"
            );
            let mut restored = Tuning {
                parameters: vec![parameter(8.0)],
                dispatched: 0,
                finished: 0,
                outcome: Outcome::default(),
                failure: None,
            };
            restored
                .restore(&tuning.render())
                .expect("rendered state parses");
            assert_eq!(restored.finished, 1);
            assert!((restored.parameters[0].value - tuning.parameters[0].value).abs() < 1e-3);
        }
    }

    #[test]
    fn logs_trajectory_at_every_interval() {
        let cases = [
            (200, 4, 200, true),
            (204, 4, 200, false),
            (196, 4, 200, false),
            (202, 4, 200, true),
            (400, 8, 200, true),
            (3, 4, 200, false),
            (5, 1, 1, true),
        ];

        for (finished, batch, every, expected) in cases {
            let tuning = Tuning {
                parameters: vec![parameter(8.0)],
                dispatched: finished,
                finished,
                outcome: Outcome::default(),
                failure: None,
            };
            assert_eq!(
                tuning.reached_log_point(batch, every),
                expected,
                "finished {finished}, batch {batch}, every {every}"
            );
        }

        let mut tuning = Tuning {
            parameters: vec![parameter(8.0), parameter(0.5)],
            dispatched: 40,
            finished: 40,
            outcome: Outcome::default(),
            failure: None,
        };
        tuning.parameters[1].name = String::from("depth_scale");
        tuning.parameters[1].value = 2.345;
        assert_eq!(tuning.trajectory_header(), "pairs,margin,depth_scale\n");
        assert_eq!(tuning.trajectory_row(), "40,100.00,2.35\n");
    }
}
