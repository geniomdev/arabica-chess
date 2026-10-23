mod checkpoint;
mod model;
mod quantize;

use std::f64::consts::PI;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Instant;

use crate::cli::{default_threads, parse_argument, usage_error};
use crate::nnue::data::{RECORD_BYTES, Record};
use crate::progress::{Ticker, bar, eta, format_duration, percent};
use crate::random::Random;
use checkpoint::Checkpoint;
use model::{Adam, Model, PARAMETERS};
use quantize::{NETWORK_EXTENSION, QuantizationError, export, network_path, quantization_error};

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

type RawRecord = [u8; RECORD_BYTES];

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn load_records(path: &Path) -> io::Result<Vec<RawRecord>> {
    let mut file = File::open(path)?;
    let bytes = file.metadata()?.len() as usize;
    if !bytes.is_multiple_of(RECORD_BYTES) {
        return Err(invalid_data(format!(
            "{} is {bytes} bytes, not a whole number of {RECORD_BYTES}-byte records",
            path.display()
        )));
    }
    let mut records = vec![[0; RECORD_BYTES]; bytes / RECORD_BYTES];
    file.read_exact(records.as_flattened_mut())?;
    if let Some(index) = records
        .iter()
        .position(|bytes| Record::from_bytes(bytes).is_none())
    {
        return Err(invalid_data(format!(
            "{}: record {index} is malformed",
            path.display()
        )));
    }
    Ok(records)
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

fn train_batch(model: &Model, batch: &[RawRecord], gradients: &mut [Vec<f32>]) -> f64 {
    let chunks: Vec<&[RawRecord]> = batch
        .chunks(chunk_size(batch.len(), gradients.len()))
        .collect();
    let loss = thread::scope(|scope| {
        let workers: Vec<_> = chunks
            .iter()
            .zip(gradients.iter_mut())
            .map(|(&records, gradient)| {
                scope.spawn(move || {
                    gradient.fill(0.0);
                    records
                        .iter()
                        .map(|bytes| {
                            f64::from(model.backpropagate(&decode(bytes), WDL_WEIGHT, gradient))
                        })
                        .sum::<f64>()
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("training worker finished"))
            .sum::<f64>()
    });
    let (total, others) = gradients.split_first_mut().expect("at least one thread");
    for other in &others[..chunks.len() - 1] {
        for (slot, value) in total.iter_mut().zip(other) {
            *slot += value;
        }
    }
    loss
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
        let rate = finished as f64 / elapsed.as_secs_f64().max(f64::EPSILON);
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
    let mut gradients = vec![vec![0.0; PARAMETERS]; run.threads];
    let mut ticker = Ticker::new();
    let mut losses = (0.0, mean_loss(&checkpoint.model, validation, run.threads));
    for epoch in run.first_epoch..run.epochs {
        let epoch_started = Instant::now();
        let rate = learning_rate(epoch, run.epochs);
        Random::new(SHUFFLE_SEED ^ epoch as u64).shuffle(training);
        let mut loss_sum = 0.0;
        let mut done = 0;
        for batch in training.chunks(BATCH_SIZE) {
            loss_sum += train_batch(&checkpoint.model, batch, &mut gradients);
            checkpoint.adam.update(
                &mut checkpoint.model,
                &gradients[0],
                rate,
                1.0 / batch.len() as f32,
            );
            done += batch.len();
            if ticker.due() {
                let line = run.progress_line(epoch, done, training.len(), loss_sum / done as f64);
                ticker.show(&line);
            }
        }
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
    let processed = ((run.epochs - run.first_epoch) * training_positions) as f64;
    let rate = processed / elapsed.as_secs_f64().max(f64::EPSILON);
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

pub fn export_command(arguments: &[String]) -> io::Result<()> {
    let [checkpoint_path, rest @ ..] = arguments else {
        return Err(usage_error("arabica export <checkpoint> [network]"));
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

pub fn run(arguments: &[String]) -> io::Result<()> {
    let [data, checkpoint_path, rest @ ..] = arguments else {
        return Err(usage_error(
            "arabica train <data> <checkpoint> [epochs] [threads]",
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
    let mut records = load_records(Path::new(data))?;
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
    use crate::board::testing::board;
    use crate::board::{KIWIPETE, START_POSITION};
    use crate::nnue::data::Outcome;

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

    #[test]
    fn threaded_batches_match_single_thread_gradients() {
        let fens = [
            (START_POSITION, 20, Outcome::Draw),
            (KIWIPETE, -90, Outcome::Loss),
            ("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1", 150, Outcome::Win),
            ("6k1/5p1p/4N3/8/8/8/8/4K1Q1 b - - 0 1", -800, Outcome::Loss),
            (
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                40,
                Outcome::Draw,
            ),
        ];
        let batch: Vec<RawRecord> = fens
            .iter()
            .map(|&(fen, score, outcome)| Record::new(&board(fen), score, outcome).to_bytes())
            .collect();
        let model = Model::random(&mut Random::new(5));
        let mut single = vec![vec![0.0; PARAMETERS]];
        let single_loss = train_batch(&model, &batch, &mut single);
        let thread_counts = [2, 3, 8];

        for threads in thread_counts {
            let mut gradients = vec![vec![1.0; PARAMETERS]; threads];
            let loss = train_batch(&model, &batch, &mut gradients);
            assert!((loss - single_loss).abs() < 1e-6, "{threads} threads");
            let largest_difference = gradients[0]
                .iter()
                .zip(&single[0])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(
                largest_difference < 1e-6,
                "{threads} threads, difference {largest_difference}"
            );
        }
    }

    #[test]
    fn loads_only_whole_valid_record_files() {
        let valid = Record::new(&board(START_POSITION), 0, Outcome::Draw).to_bytes();
        let mut broken = valid;
        broken[26] = 9;
        let cases: [(&str, Vec<u8>, Option<usize>); 4] = [
            ("empty", Vec::new(), Some(0)),
            ("two", [valid, valid].concat(), Some(2)),
            ("partial", valid[..31].to_vec(), None),
            ("broken", [valid, broken].concat(), None),
        ];

        for (name, bytes, expected) in cases {
            let path =
                std::env::temp_dir().join(format!("arabica-{}-{name}.bin", std::process::id()));
            std::fs::write(&path, bytes).expect("file written");
            let loaded = load_records(&path).ok().map(|records| records.len());
            std::fs::remove_file(&path).expect("file removed");
            assert_eq!(loaded, expected, "{name}");
        }
    }
}
