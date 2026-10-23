use arabica::nnue::data::Record;
use arabica::nnue::{HIDDEN, INPUTS, SCALE};
use arabica::random::Random;
use arabica::types::{Color, SIDES};

pub const FEATURE_BIAS: usize = INPUTS * HIDDEN;
pub const OUTPUT_WEIGHTS: usize = FEATURE_BIAS + HIDDEN;
pub const OUTPUT_BIAS: usize = OUTPUT_WEIGHTS + SIDES * HIDDEN;
pub const PARAMETERS: usize = OUTPUT_BIAS + 1;
pub const WEIGHT_LIMIT: f32 = 1.98;

const MAX_ACTIVE_FEATURES: usize = 32;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPSILON: f32 = 1e-8;

pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub fn target(record: &Record, wdl_weight: f32) -> f32 {
    let wdl = f32::from(record.outcome as u8) / 2.0;
    let score = sigmoid(f32::from(record.score) / SCALE as f32);
    wdl_weight * wdl + (1.0 - wdl_weight) * score
}

fn screlu(value: f32) -> f32 {
    let clipped = value.clamp(0.0, 1.0);
    clipped * clipped
}

fn screlu_slope(value: f32) -> f32 {
    if value > 0.0 && value < 1.0 {
        2.0 * value
    } else {
        0.0
    }
}

struct Features {
    indices: [usize; MAX_ACTIVE_FEATURES],
    len: usize,
}

impl Features {
    fn of(record: &Record, perspective: Color) -> Self {
        let mut features = Self {
            indices: [0; MAX_ACTIVE_FEATURES],
            len: 0,
        };
        for index in record.features(perspective) {
            features.indices[features.len] = index;
            features.len += 1;
        }
        features
    }

    fn as_slice(&self) -> &[usize] {
        &self.indices[..self.len]
    }
}

struct Activations {
    features: [Features; SIDES],
    accumulators: [[f32; HIDDEN]; SIDES],
    output: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub parameters: Vec<f32>,
}

impl Model {
    pub fn random(random: &mut Random) -> Self {
        let mut uniform = |bound: f32| (2.0 * random.unit() - 1.0) * bound;
        let feature_bound = 1.0 / (INPUTS as f32).sqrt();
        let output_bound = 1.0 / ((SIDES * HIDDEN) as f32).sqrt();
        let mut parameters = vec![0.0; PARAMETERS];
        for weight in &mut parameters[..FEATURE_BIAS] {
            *weight = uniform(feature_bound);
        }
        for weight in &mut parameters[OUTPUT_WEIGHTS..OUTPUT_BIAS] {
            *weight = uniform(output_bound);
        }
        Self { parameters }
    }

    fn forward(&self, record: &Record) -> Activations {
        let features = [
            Features::of(record, Color::White),
            Features::of(record, Color::Black),
        ];
        let mut accumulators = [[0.0; HIDDEN]; SIDES];
        let bias = &self.parameters[FEATURE_BIAS..OUTPUT_WEIGHTS];
        let mut output = self.parameters[OUTPUT_BIAS];
        for (side, accumulator) in accumulators.iter_mut().enumerate() {
            accumulator.copy_from_slice(bias);
            for &feature in features[side].as_slice() {
                let row = &self.parameters[feature * HIDDEN..(feature + 1) * HIDDEN];
                for (value, weight) in accumulator.iter_mut().zip(row) {
                    *value += weight;
                }
            }
            let weights = &self.parameters
                [OUTPUT_WEIGHTS + side * HIDDEN..OUTPUT_WEIGHTS + (side + 1) * HIDDEN];
            output += accumulator
                .iter()
                .zip(weights)
                .map(|(&value, weight)| screlu(value) * weight)
                .sum::<f32>();
        }
        Activations {
            features,
            accumulators,
            output,
        }
    }

    pub fn output(&self, record: &Record) -> f32 {
        self.forward(record).output
    }

    pub fn loss(&self, record: &Record, wdl_weight: f32) -> f32 {
        let error = sigmoid(self.output(record)) - target(record, wdl_weight);
        error * error
    }

    pub fn backpropagate(&self, record: &Record, wdl_weight: f32, gradient: &mut [f32]) -> f32 {
        let activations = self.forward(record);
        let prediction = sigmoid(activations.output);
        let error = prediction - target(record, wdl_weight);
        let output_gradient = 2.0 * error * prediction * (1.0 - prediction);

        gradient[OUTPUT_BIAS] += output_gradient;
        for side in 0..SIDES {
            let accumulator = &activations.accumulators[side];
            let output_offset = OUTPUT_WEIGHTS + side * HIDDEN;
            let mut hidden_gradient = [0.0; HIDDEN];
            for (index, &value) in accumulator.iter().enumerate() {
                gradient[output_offset + index] += output_gradient * screlu(value);
                hidden_gradient[index] =
                    output_gradient * self.parameters[output_offset + index] * screlu_slope(value);
            }
            for (slot, &value) in gradient[FEATURE_BIAS..OUTPUT_WEIGHTS]
                .iter_mut()
                .zip(&hidden_gradient)
            {
                *slot += value;
            }
            for &feature in activations.features[side].as_slice() {
                let row = &mut gradient[feature * HIDDEN..(feature + 1) * HIDDEN];
                for (slot, &value) in row.iter_mut().zip(&hidden_gradient) {
                    *slot += value;
                }
            }
        }
        error * error
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Adam {
    pub step: u64,
    pub first_moments: Vec<f32>,
    pub second_moments: Vec<f32>,
}

impl Adam {
    pub fn new() -> Self {
        Self {
            step: 0,
            first_moments: vec![0.0; PARAMETERS],
            second_moments: vec![0.0; PARAMETERS],
        }
    }

    pub fn update(&mut self, model: &mut Model, gradient: &[f32], learning_rate: f32, scale: f32) {
        self.step += 1;
        let correction1 = 1.0 - BETA1.powi(self.step.min(i32::MAX as u64) as i32);
        let correction2 = 1.0 - BETA2.powi(self.step.min(i32::MAX as u64) as i32);
        for (((parameter, &raw), first), second) in model
            .parameters
            .iter_mut()
            .zip(gradient)
            .zip(&mut self.first_moments)
            .zip(&mut self.second_moments)
        {
            let grad = raw * scale;
            *first = BETA1 * *first + (1.0 - BETA1) * grad;
            *second = BETA2 * *second + (1.0 - BETA2) * grad * grad;
            let step =
                learning_rate * (*first / correction1) / ((*second / correction2).sqrt() + EPSILON);
            *parameter = (*parameter - step).clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::board;
    use arabica::board::{KIWIPETE, START_POSITION};
    use arabica::nnue::data::Outcome;
    use arabica::nnue::feature_index;
    use arabica::types::{Piece, Square};

    fn records() -> Vec<Record> {
        let cases = [
            (START_POSITION, 30, Outcome::Draw),
            (KIWIPETE, -150, Outcome::Loss),
            ("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1", -250, Outcome::Loss),
            ("6k1/5p1p/4N3/8/8/8/8/4K1Q1 w - - 0 1", 900, Outcome::Win),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", 10, Outcome::Draw),
            (
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                60,
                Outcome::Win,
            ),
        ];
        cases
            .iter()
            .map(|&(fen, score, outcome)| Record::new(&board(fen), score, outcome))
            .collect()
    }

    fn total_loss(model: &Model, records: &[Record]) -> f64 {
        records
            .iter()
            .map(|record| f64::from(model.loss(record, 0.25)))
            .sum()
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let records = records();
        let mut model = Model::random(&mut Random::new(7));
        for bias in &mut model.parameters[FEATURE_BIAS..OUTPUT_WEIGHTS] {
            *bias = 0.3;
        }
        let our_king = feature_index(Color::White, Color::White, Piece::King, Square::E1);
        let their_queen = feature_index(Color::Black, Color::Black, Piece::Queen, Square::D8);
        let parameters = [
            our_king * HIDDEN,
            our_king * HIDDEN + 77,
            their_queen * HIDDEN + 5,
            FEATURE_BIAS + 3,
            FEATURE_BIAS + 120,
            OUTPUT_WEIGHTS + 9,
            OUTPUT_WEIGHTS + HIDDEN + 64,
            OUTPUT_BIAS,
        ];

        let mut gradient = vec![0.0; PARAMETERS];
        for record in &records {
            model.backpropagate(record, 0.25, &mut gradient);
        }
        let step = 1e-3;
        for parameter in parameters {
            let original = model.parameters[parameter];
            model.parameters[parameter] = original + step;
            let above = total_loss(&model, &records);
            model.parameters[parameter] = original - step;
            let below = total_loss(&model, &records);
            model.parameters[parameter] = original;
            let numeric = (above - below) / (2.0 * f64::from(step));
            let analytic = f64::from(gradient[parameter]);
            let tolerance = 1e-4 + 0.02 * numeric.abs().max(analytic.abs());
            assert!(
                (numeric - analytic).abs() <= tolerance,
                "parameter {parameter}: numeric {numeric}, analytic {analytic}"
            );
        }
    }

    #[test]
    fn adam_fits_a_handful_of_positions() {
        let records = records();
        let mut model = Model::random(&mut Random::new(11));
        let mut adam = Adam::new();
        let initial = total_loss(&model, &records);
        for _ in 0..300 {
            let mut gradient = vec![0.0; PARAMETERS];
            for record in &records {
                model.backpropagate(record, 0.25, &mut gradient);
            }
            adam.update(&mut model, &gradient, 0.01, 1.0 / records.len() as f32);
        }
        let fitted = total_loss(&model, &records);
        assert!(
            fitted < initial * 0.01,
            "initial {initial}, fitted {fitted}"
        );
        assert!(
            model
                .parameters
                .iter()
                .all(|weight| weight.abs() <= WEIGHT_LIMIT),
            "weights escaped the clipping range"
        );
    }

    #[test]
    fn targets_blend_score_and_outcome() {
        let position = board(START_POSITION);
        let cases = [
            (0, Outcome::Draw, 0.0, 0.5),
            (0, Outcome::Win, 1.0, 1.0),
            (0, Outcome::Win, 0.5, 0.75),
            (400, Outcome::Loss, 0.0, sigmoid(1.0)),
            (-400, Outcome::Loss, 0.25, 0.75 * sigmoid(-1.0)),
        ];

        for (score, outcome, wdl_weight, expected) in cases {
            let record = Record::new(&position, score, outcome);
            let value = target(&record, wdl_weight);
            assert!(
                (value - expected).abs() < 1e-6,
                "score {score}, {outcome:?}, weight {wdl_weight}: {value}"
            );
        }
    }
}
