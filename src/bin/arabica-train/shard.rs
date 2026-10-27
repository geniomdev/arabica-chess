use std::hint::spin_loop;
use std::ops::Range;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicUsize, Ordering};
use std::thread;

use arabica::nnue::data::{RECORD_BYTES, Record};
use arabica::nnue::{HIDDEN, INPUTS};
use arabica::types::{Color, SIDES};

use super::model::{
    Adam, AdamStep, MAX_ACTIVE_FEATURES, Model, OUTPUT_BIAS, screlu, screlu_slope, sigmoid, target,
};

pub type RawRecord = [u8; RECORD_BYTES];

pub const COLUMN_BLOCK: usize = 16;

const BLOCKS: usize = HIDDEN / COLUMN_BLOCK;
const ROWS: usize = INPUTS + 1 + SIDES;
const BIAS_ROW: usize = INPUTS;
const OUTPUT_ROW: usize = INPUTS + 1;
const CHUNK: usize = 256;
const SPINS_BEFORE_YIELD: u32 = 1 << 14;
const TILES: [usize; 3] = [64, 32, COLUMN_BLOCK];

const _: () = assert!(HIDDEN.is_multiple_of(COLUMN_BLOCK));

pub fn column_ranges(threads: usize) -> Vec<Range<usize>> {
    let shards = threads.clamp(1, BLOCKS);
    (0..shards)
        .map(|shard| {
            let blocks = share(BLOCKS, shard, shards);
            blocks.start * COLUMN_BLOCK..blocks.end * COLUMN_BLOCK
        })
        .collect()
}

fn tiles(width: usize) -> impl Iterator<Item = (usize, usize)> {
    let mut start = 0;
    std::iter::from_fn(move || {
        let size = TILES.into_iter().find(|&size| start + size <= width)?;
        start += size;
        Some((start - size, size))
    })
}

fn tile_at<const TILE: usize>(values: &[f32], start: usize) -> &[f32; TILE] {
    values[start..start + TILE]
        .try_into()
        .expect("tile lies inside the row")
}

fn tile_at_mut<const TILE: usize>(values: &mut [f32], start: usize) -> &mut [f32; TILE] {
    (&mut values[start..start + TILE])
        .try_into()
        .expect("tile lies inside the row")
}

struct Layer<'a> {
    parameters: &'a [f32],
    width: usize,
    output_row: usize,
    features: &'a [u16],
}

impl Layer<'_> {
    fn forward<const TILE: usize>(&self, tile: usize, accumulators: &mut [f32]) -> f32 {
        let width = self.width;
        let mut sums = *tile_at::<TILE>(self.parameters, BIAS_ROW * width + tile);
        for &feature in self.features {
            let row = tile_at::<TILE>(self.parameters, usize::from(feature) * width + tile);
            for (sum, weight) in sums.iter_mut().zip(row) {
                *sum += weight;
            }
        }
        *tile_at_mut::<TILE>(accumulators, tile) = sums;
        let weights = tile_at::<TILE>(self.parameters, self.output_row + tile);
        let mut partials = [0.0; TILE];
        for ((partial, &sum), weight) in partials.iter_mut().zip(&sums).zip(weights) {
            *partial = screlu(sum) * weight;
        }
        partials.iter().sum()
    }

    fn backward<const TILE: usize>(
        &self,
        tile: usize,
        accumulators: &[f32],
        output_gradient: f32,
        gradient: &mut [f32],
    ) {
        let width = self.width;
        let sums = tile_at::<TILE>(accumulators, tile);
        let weights = tile_at::<TILE>(self.parameters, self.output_row + tile);
        let mut hidden = [0.0; TILE];
        for ((slot, &sum), &weight) in hidden.iter_mut().zip(sums).zip(weights) {
            *slot = output_gradient * weight * screlu_slope(sum);
        }
        for (slot, &sum) in tile_at_mut::<TILE>(gradient, self.output_row + tile)
            .iter_mut()
            .zip(sums)
        {
            *slot += output_gradient * screlu(sum);
        }
        for (slot, value) in tile_at_mut::<TILE>(gradient, BIAS_ROW * width + tile)
            .iter_mut()
            .zip(&hidden)
        {
            *slot += value;
        }
        for &feature in self.features {
            let row = tile_at_mut::<TILE>(gradient, usize::from(feature) * width + tile);
            for (slot, value) in row.iter_mut().zip(&hidden) {
                *slot += value;
            }
        }
    }
}

fn columns_of(global: &[f32], columns: &Range<usize>) -> Vec<f32> {
    let mut local: Vec<f32> = (0..ROWS)
        .flat_map(|row| &global[row * HIDDEN + columns.start..row * HIDDEN + columns.end])
        .copied()
        .collect();
    local.push(global[OUTPUT_BIAS]);
    local
}

fn write_columns(local: &[f32], columns: &Range<usize>, global: &mut [f32]) {
    let width = columns.len();
    for (row, values) in local[..ROWS * width].chunks_exact(width).enumerate() {
        global[row * HIDDEN + columns.start..row * HIDDEN + columns.end].copy_from_slice(values);
    }
    global[OUTPUT_BIAS] = local[ROWS * width];
}

pub struct Shard {
    columns: Range<usize>,
    parameters: Vec<f32>,
    gradient: Vec<f32>,
    first_moments: Vec<f32>,
    second_moments: Vec<f32>,
}

impl Shard {
    pub fn split(model: &Model, adam: &Adam, threads: usize) -> Vec<Self> {
        column_ranges(threads)
            .into_iter()
            .map(|columns| {
                let parameters = columns_of(&model.parameters, &columns);
                Self {
                    gradient: vec![0.0; parameters.len()],
                    first_moments: columns_of(&adam.first_moments, &columns),
                    second_moments: columns_of(&adam.second_moments, &columns),
                    parameters,
                    columns,
                }
            })
            .collect()
    }

    pub fn merge(shards: &[Self], model: &mut Model, adam: &mut Adam) {
        for shard in shards {
            write_columns(&shard.parameters, &shard.columns, &mut model.parameters);
            write_columns(
                &shard.first_moments,
                &shard.columns,
                &mut adam.first_moments,
            );
            write_columns(
                &shard.second_moments,
                &shard.columns,
                &mut adam.second_moments,
            );
        }
    }

    fn width(&self) -> usize {
        self.columns.len()
    }

    fn output_bias_index(&self) -> usize {
        ROWS * self.width()
    }

    fn forward(&self, position: &Position, cache: &mut [f32]) -> f32 {
        let width = self.width();
        let mut partial = 0.0;
        for (side, accumulators) in cache.chunks_exact_mut(width).enumerate() {
            let layer = Layer {
                parameters: &self.parameters,
                width,
                output_row: (OUTPUT_ROW + side) * width,
                features: position.side(side),
            };
            for (start, size) in tiles(width) {
                partial += match size {
                    64 => layer.forward::<64>(start, accumulators),
                    32 => layer.forward::<32>(start, accumulators),
                    _ => layer.forward::<COLUMN_BLOCK>(start, accumulators),
                };
            }
        }
        partial
    }

    fn backward(&mut self, position: &Position, cache: &[f32], output_gradient: f32) {
        let width = self.width();
        let bias_index = self.output_bias_index();
        self.gradient[bias_index] += output_gradient;
        for (side, accumulators) in cache.chunks_exact(width).enumerate() {
            let layer = Layer {
                parameters: &self.parameters,
                width,
                output_row: (OUTPUT_ROW + side) * width,
                features: position.side(side),
            };
            for (start, size) in tiles(width) {
                let gradient = &mut self.gradient;
                match size {
                    64 => layer.backward::<64>(start, accumulators, output_gradient, gradient),
                    32 => layer.backward::<32>(start, accumulators, output_gradient, gradient),
                    _ => layer.backward::<COLUMN_BLOCK>(
                        start,
                        accumulators,
                        output_gradient,
                        gradient,
                    ),
                }
            }
        }
    }

    fn output_bias(&self) -> f32 {
        self.parameters[self.output_bias_index()]
    }

    fn update(&mut self, step: &AdamStep) {
        step.apply(
            &mut self.parameters,
            &self.gradient,
            &mut self.first_moments,
            &mut self.second_moments,
        );
        self.gradient.fill(0.0);
    }
}

#[derive(Clone, Copy)]
struct Position {
    features: [[u16; MAX_ACTIVE_FEATURES]; SIDES],
    count: usize,
    target: f32,
}

impl Position {
    const EMPTY: Self = Self {
        features: [[0; MAX_ACTIVE_FEATURES]; SIDES],
        count: 0,
        target: 0.0,
    };

    fn decode(bytes: &RawRecord, wdl_weight: f32) -> Self {
        let record = Record::from_bytes(bytes).expect("records are validated on load");
        let mut position = Self::EMPTY;
        for (features, perspective) in position.features.iter_mut().zip(Color::ALL) {
            for (slot, feature) in features.iter_mut().zip(record.features(perspective)) {
                *slot = feature as u16;
            }
        }
        position.count = record.piece_count();
        position.target = target(&record, wdl_weight);
        position
    }

    fn side(&self, side: usize) -> &[u16] {
        &self.features[side][..self.count]
    }
}

struct Exchange {
    shards: usize,
    batch: usize,
    features: Vec<AtomicU16>,
    counts: Vec<AtomicU8>,
    targets: Vec<AtomicU32>,
    partials: Vec<AtomicU32>,
}

impl Exchange {
    fn new(shards: usize, batch: usize) -> Self {
        Self {
            shards,
            batch,
            features: atomics(batch * SIDES * MAX_ACTIVE_FEATURES),
            counts: atomics(batch),
            targets: atomics(batch),
            partials: atomics(shards * batch),
        }
    }

    fn store(&self, index: usize, position: &Position) {
        let slots = &self.features[index * SIDES * MAX_ACTIVE_FEATURES..];
        for (slot, &feature) in slots.iter().zip(position.features.as_flattened()) {
            slot.store(feature, Ordering::Relaxed);
        }
        self.counts[index].store(position.count as u8, Ordering::Relaxed);
        self.targets[index].store(position.target.to_bits(), Ordering::Relaxed);
    }

    fn load(&self, index: usize) -> Position {
        let mut position = Position::EMPTY;
        position.count = usize::from(self.counts[index].load(Ordering::Relaxed));
        position.target = f32::from_bits(self.targets[index].load(Ordering::Relaxed));
        for (side, features) in position.features.iter_mut().enumerate() {
            let slots = &self.features[(index * SIDES + side) * MAX_ACTIVE_FEATURES..];
            for (feature, slot) in features.iter_mut().zip(slots).take(position.count) {
                *feature = slot.load(Ordering::Relaxed);
            }
        }
        position
    }

    fn publish(&self, shard: usize, index: usize, partial: f32) {
        self.partials[shard * self.batch + index].store(partial.to_bits(), Ordering::Relaxed);
    }

    fn output(&self, index: usize) -> f32 {
        (0..self.shards)
            .map(|shard| {
                f32::from_bits(self.partials[shard * self.batch + index].load(Ordering::Relaxed))
            })
            .sum()
    }
}

struct SpinBarrier {
    parties: usize,
    arrived: AtomicUsize,
    generation: AtomicUsize,
}

impl SpinBarrier {
    fn new(parties: usize) -> Self {
        Self {
            parties,
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
        }
    }

    fn wait(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == self.parties {
            self.arrived.store(0, Ordering::Relaxed);
            self.generation
                .store(generation.wrapping_add(1), Ordering::Release);
            return;
        }
        let mut spins = 0;
        while self.generation.load(Ordering::Acquire) == generation {
            if spins < SPINS_BEFORE_YIELD {
                spins += 1;
                spin_loop();
            } else {
                thread::yield_now();
            }
        }
    }
}

pub struct Epoch<'a> {
    pub records: &'a [RawRecord],
    pub batch_size: usize,
    pub learning_rate: f32,
    pub first_step: u64,
    pub wdl_weight: f32,
}

struct Context<'a> {
    epoch: &'a Epoch<'a>,
    exchange: Exchange,
    barrier: SpinBarrier,
}

fn atomics<T: Default>(len: usize) -> Vec<T> {
    (0..len).map(|_| T::default()).collect()
}

fn share(len: usize, shard: usize, shards: usize) -> Range<usize> {
    len * shard / shards..len * (shard + 1) / shards
}

fn work(context: &Context, index: usize, shard: &mut Shard, mut report: impl FnMut(usize, f64)) {
    let Epoch {
        records,
        batch_size,
        learning_rate,
        first_step,
        wdl_weight,
    } = *context.epoch;
    let exchange = &context.exchange;
    let mut cache = vec![0.0; CHUNK * SIDES * shard.width()];
    let mut positions = [Position::EMPTY; CHUNK];
    let mut loss = 0.0;
    let mut done = 0;
    for (batch_index, batch) in records.chunks(batch_size).enumerate() {
        for position in share(batch.len(), index, exchange.shards) {
            exchange.store(position, &Position::decode(&batch[position], wdl_weight));
        }
        context.barrier.wait();
        for chunk_start in (0..batch.len()).step_by(CHUNK) {
            let chunk = chunk_start..batch.len().min(chunk_start + CHUNK);
            for (offset, position) in chunk.clone().enumerate() {
                positions[offset] = exchange.load(position);
                let cached = &mut cache[offset * SIDES * shard.width()..][..SIDES * shard.width()];
                exchange.publish(index, position, shard.forward(&positions[offset], cached));
            }
            context.barrier.wait();
            for (offset, position) in chunk.enumerate() {
                let prediction = sigmoid(exchange.output(position) + shard.output_bias());
                let error = prediction - positions[offset].target;
                loss += f64::from(error * error);
                let cached = &cache[offset * SIDES * shard.width()..][..SIDES * shard.width()];
                let output_gradient = 2.0 * error * prediction * (1.0 - prediction);
                shard.backward(&positions[offset], cached, output_gradient);
            }
        }
        let step = first_step + batch_index as u64 + 1;
        shard.update(&AdamStep::new(
            step,
            learning_rate,
            1.0 / batch.len() as f32,
        ));
        done += batch.len();
        report(done, loss);
        context.barrier.wait();
    }
}

pub fn train_epoch(epoch: &Epoch, shards: &mut [Shard], report: impl FnMut(usize, f64)) -> f64 {
    let context = Context {
        epoch,
        exchange: Exchange::new(shards.len(), epoch.batch_size),
        barrier: SpinBarrier::new(shards.len()),
    };
    let mut total = 0.0;
    let (first, rest) = shards.split_first_mut().expect("at least one shard");
    thread::scope(|scope| {
        for (offset, shard) in rest.iter_mut().enumerate() {
            let context = &context;
            scope.spawn(move || work(context, offset + 1, shard, |_, _| {}));
        }
        let mut report = report;
        work(&context, 0, first, |done, loss| {
            total = loss;
            report(done, loss);
        });
    });
    total
}

pub fn batches(records: usize, batch_size: usize) -> u64 {
    records.div_ceil(batch_size) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::PARAMETERS;
    use crate::testing::board;
    use arabica::board::{KIWIPETE, START_POSITION};
    use arabica::nnue::data::Outcome;
    use arabica::random::Random;

    #[test]
    fn column_ranges_tile_the_hidden_layer() {
        let cases = [
            (1, 1),
            (2, 2),
            (5, 5),
            (12, 12),
            (BLOCKS, BLOCKS),
            (100, BLOCKS),
        ];

        for (threads, expected) in cases {
            let ranges = column_ranges(threads);
            assert_eq!(ranges.len(), expected, "{threads} threads");
            assert_eq!(ranges[0].start, 0, "{threads} threads");
            assert_eq!(ranges[expected - 1].end, HIDDEN, "{threads} threads");
            for pair in ranges.windows(2) {
                assert_eq!(pair[0].end, pair[1].start, "{threads} threads");
            }
            for range in &ranges {
                assert!(!range.is_empty(), "{threads} threads");
                assert!(
                    range.len().is_multiple_of(COLUMN_BLOCK),
                    "{threads} threads"
                );
                assert!(
                    range.len() <= ranges[0].len() + COLUMN_BLOCK,
                    "{threads} threads"
                );
            }
        }
    }

    fn spread(random: &mut Random) -> Vec<f32> {
        (0..PARAMETERS).map(|_| random.unit() - 0.5).collect()
    }

    #[test]
    fn shards_round_trip_models_and_moments() {
        let mut random = Random::new(9);
        let model = Model {
            parameters: spread(&mut random),
        };
        let mut adam = Adam::new();
        adam.first_moments = spread(&mut random);
        adam.second_moments = spread(&mut random);
        let thread_counts = [1, 3, 12, BLOCKS];

        for threads in thread_counts {
            let shards = Shard::split(&model, &adam, threads);
            let mut merged_model = Model {
                parameters: vec![0.0; PARAMETERS],
            };
            let mut merged_adam = Adam::new();
            Shard::merge(&shards, &mut merged_model, &mut merged_adam);
            assert_eq!(merged_model, model, "{threads} threads");
            assert_eq!(
                merged_adam.first_moments, adam.first_moments,
                "{threads} threads"
            );
            assert_eq!(
                merged_adam.second_moments, adam.second_moments,
                "{threads} threads"
            );
        }
    }

    fn records() -> Vec<RawRecord> {
        let cases = [
            (START_POSITION, 20, Outcome::Draw),
            (KIWIPETE, -90, Outcome::Loss),
            ("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1", 150, Outcome::Win),
            ("6k1/5p1p/4N3/8/8/8/8/4K1Q1 b - - 0 1", -800, Outcome::Loss),
            (
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                40,
                Outcome::Draw,
            ),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", 10, Outcome::Draw),
            ("8/8/8/8/8/8/8/k6K w - - 0 1", 0, Outcome::Draw),
        ];
        cases
            .iter()
            .map(|&(fen, score, outcome)| Record::new(&board(fen), score, outcome).to_bytes())
            .collect()
    }

    fn reference_epoch(records: &[RawRecord], batch_size: usize) -> (Model, f64) {
        let mut model = Model::random(&mut Random::new(5));
        let mut adam = Adam::new();
        let mut loss = 0.0;
        for batch in records.chunks(batch_size) {
            let mut gradient = vec![0.0; PARAMETERS];
            for bytes in batch {
                let record = Record::from_bytes(bytes).expect("valid record");
                loss += f64::from(model.backpropagate(&record, 0.25, &mut gradient));
            }
            adam.update(&mut model, &gradient, 0.01, 1.0 / batch.len() as f32);
        }
        (model, loss)
    }

    #[test]
    fn sharded_epochs_match_the_reference_trainer() {
        let records = records();
        let cases = [(1, 7), (2, 7), (3, 2), (12, 3), (BLOCKS, 4), (100, 1)];

        for (threads, batch_size) in cases {
            let (expected, expected_loss) = reference_epoch(&records, batch_size);
            let mut model = Model::random(&mut Random::new(5));
            let mut adam = Adam::new();
            let mut shards = Shard::split(&model, &adam, threads);
            let epoch = Epoch {
                records: &records,
                batch_size,
                learning_rate: 0.01,
                first_step: 0,
                wdl_weight: 0.25,
            };
            let loss = train_epoch(&epoch, &mut shards, |_, _| {});
            Shard::merge(&shards, &mut model, &mut adam);
            let context = format!("{threads} threads, batch {batch_size}");
            assert!(
                (loss - expected_loss).abs() < 1e-5,
                "{context}: loss {loss}"
            );
            let largest_difference = model
                .parameters
                .iter()
                .zip(&expected.parameters)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(
                largest_difference < 1e-4,
                "{context}: difference {largest_difference}"
            );
        }
    }
}
