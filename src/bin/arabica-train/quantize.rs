use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::checkpoint::replace_file;
use super::model::{Model, OUTPUT_BIAS, OUTPUT_WEIGHTS};
use arabica::nnue::data::Record;
use arabica::nnue::{Network, QA, QB, SCALE};
use arabica::types::Color;

pub const NETWORK_EXTENSION: &str = "nnue";

fn quantized(value: f32, scale: i32) -> i16 {
    (value * scale as f32)
        .round()
        .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
}

fn scale_of(parameter: usize) -> i32 {
    match parameter {
        OUTPUT_BIAS => QA * QB,
        index if index >= OUTPUT_WEIGHTS => QB,
        _ => QA,
    }
}

pub fn quantize(model: &Model) -> Box<Network> {
    let mut network = Network::zeroed();
    for (parameter, (slot, &value)) in network.values_mut().zip(&model.parameters).enumerate() {
        *slot = quantized(value, scale_of(parameter));
    }
    network
}

pub fn network_path(checkpoint: &Path) -> PathBuf {
    checkpoint.with_extension(NETWORK_EXTENSION)
}

pub fn network_output(network: &Network, record: &Record) -> i32 {
    let us = network.refresh(record.features(Color::White));
    let them = network.refresh(record.features(Color::Black));
    network.evaluate(&us, &them)
}

pub struct QuantizationError {
    pub mean: f64,
    pub max: f64,
}

pub fn quantization_error(model: &Model, records: &[Record]) -> QuantizationError {
    let network = quantize(model);
    let errors = records.iter().map(|record| {
        let float = f64::from(model.output(record)) * f64::from(SCALE);
        (f64::from(network_output(&network, record)) - float).abs()
    });
    let (sum, max) = errors.fold((0.0, 0.0), |(sum, max): (f64, f64), error| {
        (sum + error, max.max(error))
    });
    QuantizationError {
        mean: sum / records.len().max(1) as f64,
        max,
    }
}

pub fn export(model: &Model, path: &Path) -> io::Result<()> {
    let bytes = quantize(model).to_bytes();
    replace_file(path, |writer| writer.write_all(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FEATURE_BIAS, PARAMETERS, WEIGHT_LIMIT};
    use crate::testing::board;
    use arabica::board::{KIWIPETE, START_POSITION};
    use arabica::nnue::NETWORK_BYTES;
    use arabica::nnue::data::Outcome;
    use arabica::random::Random;

    #[test]
    fn quantizes_with_rounding_and_saturation() {
        let cases = [
            (0.0, QA, 0),
            (1.0, QA, 255),
            (-1.98, QA, -505),
            (0.6 / QA as f32, QA, 1),
            (1.98, QB, 127),
            (1.98, QA * QB, 32_314),
            (3.0, QA * QB, i16::MAX),
            (-3.0, QA * QB, i16::MIN),
        ];

        for (value, scale, expected) in cases {
            assert_eq!(quantized(value, scale), expected, "{value} x {scale}");
        }
    }

    fn spread_model(seed: u64) -> Model {
        let mut random = Random::new(seed);
        let mut uniform = |low: f32, high: f32| low + (high - low) * random.unit();
        let mut parameters = vec![0.0; PARAMETERS];
        for (index, value) in parameters.iter_mut().enumerate() {
            *value = match index {
                OUTPUT_BIAS => uniform(-0.5, 0.5),
                index if index >= OUTPUT_WEIGHTS => uniform(-0.5, 0.5),
                index if index >= FEATURE_BIAS => uniform(0.0, 0.4),
                _ => uniform(-0.12, 0.12),
            };
        }
        Model { parameters }
    }

    fn on_quantization_grid(mut model: Model) -> Model {
        for (parameter, value) in model.parameters.iter_mut().enumerate() {
            let scale = scale_of(parameter) as f32;
            *value = (*value * scale).round() / scale;
        }
        model
    }

    const FENS: [&str; 7] = [
        START_POSITION,
        KIWIPETE,
        "4k3/8/8/8/4P3/8/8/4K3 b - - 0 1",
        "6k1/5p1p/4N3/8/8/8/8/4K1Q1 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",
    ];

    fn evaluation_errors(model: &Model) -> Vec<f32> {
        let network = quantize(model);
        FENS.iter()
            .map(|fen| {
                let record = Record::new(&board(fen), 0, Outcome::Draw);
                network_output(&network, &record) as f32 - model.output(&record) * SCALE as f32
            })
            .collect()
    }

    #[test]
    fn grid_aligned_models_quantize_exactly() {
        let seeds = [1, 2, 3, 4];

        for seed in seeds {
            let errors = evaluation_errors(&on_quantization_grid(spread_model(seed)));
            assert!(
                errors.iter().all(|error| error.abs() <= 1.0),
                "seed {seed}: errors {errors:?}"
            );
        }
    }

    #[test]
    fn rounding_noise_stays_small() {
        let seeds = [1, 2, 3, 4];

        for seed in seeds {
            let errors = evaluation_errors(&spread_model(seed));
            let mean = errors.iter().map(|error| error.abs()).sum::<f32>() / errors.len() as f32;
            assert!(mean <= 30.0, "seed {seed}: errors {errors:?}");
        }
    }

    #[test]
    fn clipped_weights_never_saturate() {
        let mut model = spread_model(4);
        for value in &mut model.parameters {
            *value = if *value >= 0.0 {
                WEIGHT_LIMIT
            } else {
                -WEIGHT_LIMIT
            };
        }
        let network = quantize(&model);
        assert!(
            network
                .values()
                .all(|value| value != i16::MIN && value != i16::MAX),
            "a clipped weight hit the i16 range"
        );
    }

    #[test]
    fn exports_loadable_networks_next_to_checkpoints() {
        let cases = [
            ("net1.ckpt", "net1.nnue"),
            ("data/nets/net2.ckpt", "data/nets/net2.nnue"),
            ("plain", "plain.nnue"),
        ];
        for (checkpoint, expected) in cases {
            assert_eq!(
                network_path(Path::new(checkpoint)),
                PathBuf::from(expected),
                "{checkpoint}"
            );
        }

        let model = spread_model(5);
        let path = std::env::temp_dir().join(format!("arabica-{}-export.nnue", std::process::id()));
        export(&model, &path).expect("network exported");
        let bytes = std::fs::read(&path).expect("network read");
        std::fs::remove_file(&path).expect("network removed");
        assert_eq!(bytes.len(), NETWORK_BYTES);
        let loaded = Network::from_bytes(&bytes).expect("network parsed");
        assert!(loaded.values().eq(quantize(&model).values()));
    }
}
