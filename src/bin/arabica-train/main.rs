mod checkpoint;
mod dataset;
mod model;
mod quantize;
mod shard;

use std::f64::consts::PI;
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Instant;

use arabica::cli::{default_threads, parse_argument, usage_error};
use arabica::nnue::data::Record;
use arabica::nnue::games::Filter;
use arabica::progress::{Ticker, bar, eta, format_duration, per_second, percent};
use arabica::random::Random;
use checkpoint::Checkpoint;
use model::{Adam, Model};
use quantize::{NETWORK_EXTENSION, QuantizationError, export, network_path, quantization_error};
use shard::{Epoch, RawRecord, Shard, batches, train_epoch};

const DEFAULT_EPOCHS: usize = 40;
const BATCH_SIZE: usize = 16_384;
const VALIDATION_DIVISOR: usize = 100;
const MAX_VALIDATION_POSITIONS: usize = 1_000_000;
const INITIAL_LEARNING_RATE: f64 = 0.001;
const FINAL_LEARNING_RATE: f64 = 0.00001;
const WDL_WEIGHT: f32 = 0.25;
const QUANTIZATION_SAMPLE: usize = 100_000;
const INITIAL_SEED: u64 = 0x5EED_0001;
const SPLIT_SEED: u64 = 0x5EED_0002;
const SHUFFLE_SEED: u64 = 0x5EED_0003;

const USAGE: &str = "arabica-train <train|export> ...";

fn main() -> io::Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.split_first() {
        Some((command, rest)) if command == "train" => run(rest),
        Some((command, rest)) if command == "export" => export_command(rest),
        _ => Err(usage_error(USAGE)),
    }
}

#[cfg(test)]
mod testing {
    use std::path::PathBuf;

    use arabica::board::Board;

    pub fn board(fen: &str) -> Board {
        fen.parse()
            .unwrap_or_else(|error| panic!("fen {fen:?} rejected: {error}"))
    }

    pub fn scratch_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("arabica-{}-{name}", std::process::id()))
    }
}

fn load_dataset(path: &Path, threads: usize) -> io::Result<Vec<RawRecord>> {
    let dataset = dataset::load(path, &Filter::DEFAULT, threads)?;
    if let Some(counts) = dataset.games {
        eprintln!(
            "train: {} games, {} plies, {} positions ({:.1}%) pass the filter {:?}",
            counts.games,
            counts.plies,
            dataset.records.len(),
            percent(dataset.records.len() as u64, counts.plies),
            Filter::DEFAULT
        );
    }
    Ok(dataset.records)
}

fn decode(bytes: &RawRecord) -> Record {
    Record::from_bytes(bytes).expect("records are validated on load")
}

fn learning_rate(epoch: usize, epochs: usize) -> f32 {
    let progress = epoch as f64 / epochs.saturating_sub(1).max(1) as f64;
    let cosine = 0.5 * (1.0 + (PI * progress.min(1.0)).cos());
    (FINAL_LEARNING_RATE + (INITIAL_LEARNING_RATE - FINAL_LEARNING_RATE) * cosine) as f32
}

fn validation_size(records: usize) -> usize {
    (records / VALIDATION_DIVISOR).min(MAX_VALIDATION_POSITIONS)
}

fn chunk_size(items: usize, threads: usize) -> usize {
    items.div_ceil(threads).max(1)
}

fn shuffle_segments(records: &mut [RawRecord], seed: u64, threads: usize) {
    let size = chunk_size(records.len(), threads);
    thread::scope(|scope| {
        for (index, segment) in records.chunks_mut(size).enumerate() {
            scope.spawn(move || Random::new(seed ^ ((index as u64) << 32)).shuffle(segment));
        }
    });
}

fn mean_loss(model: &Model, records: &[RawRecord], threads: usize) -> f64 {
    if records.is_empty() {
        return 0.0;
    }
    let total: f64 = thread::scope(|scope| {
        let workers: Vec<_> = records
            .chunks(chunk_size(records.len(), threads))
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|bytes| f64::from(model.loss(&decode(bytes), WDL_WEIGHT)))
                        .sum::<f64>()
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("validation worker finished"))
            .sum()
    });
    total / records.len() as f64
}

struct Run<'a> {
    checkpoint_path: &'a Path,
    epochs: usize,
    threads: usize,
    started: Instant,
    first_epoch: usize,
}

impl Run<'_> {
    fn progress_line(&self, epoch: usize, done: usize, epoch_size: usize, loss: f64) -> String {
        let total = ((self.epochs - self.first_epoch) * epoch_size) as u64;
        let finished = ((epoch - self.first_epoch) * epoch_size + done) as u64;
        let elapsed = self.started.elapsed();
        let rate = per_second(finished, elapsed);
        format!(
            "{} epoch {}/{} {:5.1}%  loss {loss:.6}  elapsed {}  eta {}  {rate:.0}/s",
            bar(done as u64, epoch_size as u64),
            epoch + 1,
            self.epochs,
            percent(done as u64, epoch_size as u64),
            format_duration(elapsed),
            eta(finished, total, elapsed)
        )
    }
}

fn train(
    run: &Run,
    checkpoint: &mut Checkpoint,
    training: &mut [RawRecord],
    validation: &[RawRecord],
) -> io::Result<(f64, f64)> {
    let mut shards = Shard::split(&checkpoint.model, &checkpoint.adam, run.threads);
    let mut ticker = Ticker::new();
    let mut losses = (0.0, mean_loss(&checkpoint.model, validation, run.threads));
    for epoch in run.first_epoch..run.epochs {
        let epoch_started = Instant::now();
        let rate = learning_rate(epoch, run.epochs);
        shuffle_segments(training, SHUFFLE_SEED ^ epoch as u64, run.threads);
        let plan = Epoch {
            records: training,
            batch_size: BATCH_SIZE,
            learning_rate: rate,
            first_step: checkpoint.adam.step,
            wdl_weight: WDL_WEIGHT,
        };
        let loss_sum = train_epoch(&plan, &mut shards, |done, loss_sum| {
            if ticker.due() {
                let line = run.progress_line(epoch, done, training.len(), loss_sum / done as f64);
                ticker.show(&line);
            }
        });
        checkpoint.adam.step += batches(training.len(), BATCH_SIZE);
        Shard::merge(&shards, &mut checkpoint.model, &mut checkpoint.adam);
        let training_loss = loss_sum / training.len().max(1) as f64;
        let validation_loss = mean_loss(&checkpoint.model, validation, run.threads);
        checkpoint.epoch = epoch + 1;
        checkpoint.save(run.checkpoint_path)?;
        export(&checkpoint.model, &network_path(run.checkpoint_path))?;
        ticker.finish(&format!(
            "epoch {}/{}  train {training_loss:.6}  validation {validation_loss:.6}  lr {rate:.6}  time {}",
            epoch + 1,
            run.epochs,
            format_duration(epoch_started.elapsed())
        ));
        losses = (training_loss, validation_loss);
    }
    Ok(losses)
}

fn print_summary(
    run: &Run,
    training_positions: usize,
    losses: (f64, f64),
    quantization: &QuantizationError,
) {
    let elapsed = run.started.elapsed();
    let processed = ((run.epochs - run.first_epoch) * training_positions) as u64;
    let rate = per_second(processed, elapsed);
    eprintln!("train finished");
    eprintln!("  time               {}", format_duration(elapsed));
    eprintln!(
        "  epochs             {} this run, {} total",
        run.epochs - run.first_epoch,
        run.epochs
    );
    eprintln!(
        "  speed              {rate:.0} positions/s ({:.0}/s per thread)",
        rate / run.threads as f64
    );
    eprintln!(
        "  loss               train {:.6}, validation {:.6}",
        losses.0, losses.1
    );
    eprintln!(
        "  quantization       mean {:.1} cp, max {:.1} cp from float on validation positions",
        quantization.mean, quantization.max
    );
    eprintln!("  checkpoint         {}", run.checkpoint_path.display());
    eprintln!(
        "  network            {}",
        network_path(run.checkpoint_path).display()
    );
}

fn reject_network_extension(checkpoint_path: &Path) -> io::Result<()> {
    if checkpoint_path.extension() == Some(NETWORK_EXTENSION.as_ref()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "checkpoint {} must not use the .{NETWORK_EXTENSION} extension of exported networks",
                checkpoint_path.display()
            ),
        ));
    }
    Ok(())
}

fn export_command(arguments: &[String]) -> io::Result<()> {
    let [checkpoint_path, rest @ ..] = arguments else {
        return Err(usage_error("arabica-train export <checkpoint> [network]"));
    };
    let checkpoint_path = PathBuf::from(checkpoint_path);
    reject_network_extension(&checkpoint_path)?;
    let output = rest
        .first()
        .map_or_else(|| network_path(&checkpoint_path), PathBuf::from);
    let checkpoint = Checkpoint::load(&checkpoint_path)?;
    export(&checkpoint.model, &output)?;
    eprintln!(
        "export: {} after epoch {} into {}",
        checkpoint_path.display(),
        checkpoint.epoch,
        output.display()
    );
    Ok(())
}

fn run(arguments: &[String]) -> io::Result<()> {
    let [data, checkpoint_path, rest @ ..] = arguments else {
        return Err(usage_error(
            "arabica-train train <data> <checkpoint> [epochs] [threads]",
        ));
    };
    let epochs: usize = parse_argument(rest.first(), "epochs", DEFAULT_EPOCHS)?.max(1);
    let threads: usize = parse_argument(rest.get(1), "threads", default_threads())?.max(1);
    let checkpoint_path = PathBuf::from(checkpoint_path);
    reject_network_extension(&checkpoint_path)?;

    let mut checkpoint = if checkpoint_path.exists() {
        let checkpoint = Checkpoint::load(&checkpoint_path)?;
        eprintln!(
            "train: resuming {} after epoch {}",
            checkpoint_path.display(),
            checkpoint.epoch
        );
        checkpoint
    } else {
        Checkpoint {
            epoch: 0,
            model: Model::random(&mut Random::new(INITIAL_SEED)),
            adam: Adam::new(),
        }
    };
    if checkpoint.epoch >= epochs {
        eprintln!(
            "train: {} already has {} of {epochs} epochs",
            checkpoint_path.display(),
            checkpoint.epoch
        );
        return Ok(());
    }

    let loading = Instant::now();
    let mut records = load_dataset(Path::new(data), threads)?;
    Random::new(SPLIT_SEED).shuffle(&mut records);
    let training_len = records.len() - validation_size(records.len());
    let (training, validation) = records.split_at_mut(training_len);
    eprintln!(
        "train: {} training and {} validation positions from {} in {}, {epochs} epochs, {threads} threads, batch {BATCH_SIZE}",
        training.len(),
        validation.len(),
        data,
        format_duration(loading.elapsed())
    );

    let run = Run {
        checkpoint_path: &checkpoint_path,
        epochs,
        threads,
        started: Instant::now(),
        first_epoch: checkpoint.epoch,
    };
    let losses = train(&run, &mut checkpoint, training, validation)?;
    let sample: Vec<Record> = validation
        .iter()
        .take(QUANTIZATION_SAMPLE)
        .map(decode)
        .collect();
    let quantization = quantization_error(&checkpoint.model, &sample);
    print_summary(&run, training.len(), losses, &quantization);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{board, scratch_path};
    use arabica::board::{KIWIPETE, START_POSITION};
    use arabica::nnue::data::{Outcome, RECORD_BYTES};

    #[test]
    fn learning_rate_decays_along_a_cosine() {
        let cases = [
            (0, 40, INITIAL_LEARNING_RATE),
            (39, 40, FINAL_LEARNING_RATE),
            (2, 5, (INITIAL_LEARNING_RATE + FINAL_LEARNING_RATE) / 2.0),
            (0, 1, INITIAL_LEARNING_RATE),
        ];

        for (epoch, epochs, expected) in cases {
            let rate = f64::from(learning_rate(epoch, epochs));
            assert!(
                (rate - expected).abs() < 1e-9,
                "epoch {epoch}/{epochs}: {rate}"
            );
        }
    }

    #[test]
    fn segment_shuffles_permute_within_segments() {
        let cases = [(0, 3), (1, 1), (10, 3), (1_000, 1), (1_000, 7), (5, 12)];

        for (len, threads) in cases {
            let records: Vec<RawRecord> = (0..len as u32)
                .map(|index| {
                    let mut record = [0; RECORD_BYTES];
                    record[..4].copy_from_slice(&index.to_le_bytes());
                    record
                })
                .collect();
            let mut shuffled = records.clone();
            shuffle_segments(&mut shuffled, 17, threads);
            let mut repeated = records.clone();
            shuffle_segments(&mut repeated, 17, threads);
            assert_eq!(shuffled, repeated, "{len} records, {threads} threads");
            let size = chunk_size(len, threads);
            for (original, segment) in records.chunks(size).zip(shuffled.chunks(size)) {
                let mut sorted = segment.to_vec();
                sorted.sort_unstable();
                let mut expected = original.to_vec();
                expected.sort_unstable();
                assert_eq!(sorted, expected, "{len} records, {threads} threads");
            }
            if len >= 1_000 {
                assert_ne!(shuffled, records, "{len} records, {threads} threads");
            }
        }
    }

    #[test]
    fn validation_takes_a_capped_share() {
        let cases = [
            (0, 0),
            (99, 0),
            (10_000, 100),
            (100_000_000, 1_000_000),
            (500_000_000, 1_000_000),
        ];

        for (records, expected) in cases {
            assert_eq!(validation_size(records), expected, "{records} records");
        }
    }

    fn records() -> Vec<RawRecord> {
        let cases = [
            (START_POSITION, 20, Outcome::Draw),
            (KIWIPETE, -90, Outcome::Loss),
            ("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1", 150, Outcome::Win),
            ("6k1/5p1p/4N3/8/8/8/8/4K1Q1 b - - 0 1", -800, Outcome::Loss),
            ("8/8/8/8/8/8/8/k6K w - - 0 1", 0, Outcome::Draw),
        ];
        cases
            .iter()
            .map(|&(fen, score, outcome)| Record::new(&board(fen), score, outcome).to_bytes())
            .collect()
    }

    #[test]
    fn checkpoints_never_take_the_network_extension() {
        let cases = [
            ("net.nnue", false),
            ("runs/net.nnue", false),
            ("run.ckpt", true),
            ("run", true),
            ("run.nnue.ckpt", true),
            ("nets.nnue/run.ckpt", true),
        ];

        for (path, accepted) in cases {
            assert_eq!(
                reject_network_extension(Path::new(path)).is_ok(),
                accepted,
                "{path}"
            );
        }
    }

    #[test]
    fn mean_loss_does_not_depend_on_threads() {
        let model = Model::random(&mut Random::new(11));
        let records = records();
        let sequential: f64 = records
            .iter()
            .map(|bytes| f64::from(model.loss(&decode(bytes), WDL_WEIGHT)))
            .sum::<f64>()
            / records.len() as f64;
        let cases = [
            (records.len(), 1, sequential),
            (records.len(), 2, sequential),
            (records.len(), 3, sequential),
            (records.len(), 64, sequential),
            (0, 4, 0.0),
        ];

        for (len, threads, expected) in cases {
            let loss = mean_loss(&model, &records[..len], threads);
            assert!(
                (loss - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "{len} records, {threads} threads: {loss} vs {expected}"
            );
        }
    }

    #[test]
    fn training_runs_resume_and_export() {
        let data = scratch_path("run-data.bin");
        let checkpoint_path = scratch_path("run.ckpt");
        let network = network_path(&checkpoint_path);
        let exported = scratch_path("run-exported.nnue");
        std::fs::write(&data, records().as_flattened()).expect("data written");
        let text = |path: &Path| path.display().to_string();
        let train_command = |epochs: &str| {
            vec![
                "train".to_owned(),
                text(&data),
                text(&checkpoint_path),
                epochs.to_owned(),
                "2".to_owned(),
            ]
        };
        let steps: [(Vec<String>, bool, usize); 6] = [
            (train_command("1"), true, 1),
            (train_command("1"), true, 1),
            (train_command("2"), true, 2),
            (
                vec!["train".to_owned(), text(&data), text(&network)],
                false,
                2,
            ),
            (
                vec!["export".to_owned(), text(&checkpoint_path), text(&exported)],
                true,
                2,
            ),
            (vec!["export".to_owned()], false, 2),
        ];

        for (arguments, succeeds, epoch) in steps {
            let result = match arguments.split_first() {
                Some((command, rest)) if command == "train" => run(rest),
                Some((_, rest)) => export_command(rest),
                None => unreachable!(),
            };
            assert_eq!(result.is_ok(), succeeds, "{arguments:?}: {result:?}");
            let saved = Checkpoint::load(&checkpoint_path).expect("checkpoint saved");
            assert_eq!(saved.epoch, epoch, "{arguments:?}");
            assert!(network.exists(), "{arguments:?}");
        }
        assert_eq!(
            std::fs::read(&exported).expect("exported network"),
            std::fs::read(&network).expect("trained network")
        );
        for path in [&data, &checkpoint_path, &network, &exported] {
            std::fs::remove_file(path).expect("scratch file removed");
        }
    }
}
