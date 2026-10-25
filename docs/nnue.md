# NNUE

Evaluation uses a SCReLU NNUE with a lazily updated accumulator stack and an AVX2 output layer, trained on self-play data with its own CPU trainer.

## Current network

| Property | Value |
|---|---|
| File | `networks/net2.nnue`, embedded into the binary at build time; the `EvalFile` UCI option swaps in another `.nnue` at runtime (e.g. `option.EvalFile=data/nets/net3.nnue` in fastchess) without rebuilding |
| Architecture | `768 -> 512 x2 -> 1`, perspective network |
| Inputs | 768 = 2 sides (own / opponent) x 6 piece types x 64 squares, squares flipped vertically for Black |
| Hidden layer | 512 neurons per perspective, shared feature weights, accumulator updated incrementally on a lazy stack |
| Activation | SCReLU (`clamp(x, 0, QA)^2`) |
| Output | Single neuron over the concatenated side-to-move and opponent accumulators (1024 inputs) |
| Parameters | 394 753 (`i16`, little-endian), 789 506 bytes |
| Quantization | QA = 255 (feature weights and biases), QB = 64 (output weights), output bias scaled by QA * QB |
| Evaluation scale | 400 (centipawns = output * 400 / (QA * QB)) |
| Inference | AVX2 `madd` output layer when available and the output weights fit, portable fallback otherwise; per-lane sums stay in `i32` and the total is widened to `i64`, since a saturated 512-wide layer exceeds `i32` |
| Memory layout | Network rows and accumulators aligned to 64 bytes, so every 1 KB row starts on a cache line |
| Special cases | Positions without enough material to mate evaluate to 0 |

Networks of a different hidden size are rejected by size: `net1.nnue` (128 neurons) no longer loads into this build.

## Network history

| Network | Data | Training | Result |
|---|---|---|---|
| `net1` (0.2.0), `768 -> 128 x2 -> 1` | 100M positions from 964 632 self-play games of arabica 0.1.3 (HCE), 5000 nodes per move, 12 threads, 12 h | 40 epochs, ~25 s each; final loss 0.005618 train, 0.005643 validation | +219.9 ± 58.3 Elo vs 0.1.3 (HCE), 100 games, 10+0.1, `8moves_v3.pgn`: +61 -5 =34 |

## Trainer

`arabica-train` keeps the float model in memory and trains it with Adam on batches of 16 384 positions, a cosine learning-rate decay from 0.001 to 0.00001 and targets blending 75% search score with 25% game outcome.

| Technique | Effect |
|---|---|
| Hidden-layer sharding | Each thread owns a contiguous range of hidden neurons (a multiple of 16 columns) with its weights, gradients and Adam moments, so gradients never need a cross-thread reduction and Adam runs in parallel |
| Cache-sized shards | With 12 threads a shard holds at most 48 columns: ~150 KB of weights plus ~150 KB of gradients, resident in the core's L2 |
| Register tiles | Forward and backward passes walk each shard in 64, 32 and 16 column tiles monomorphized at compile time, so a tile's sums stay in AVX registers across all active features |
| Shared decoding | Threads decode disjoint slices of the batch into a shared buffer once, then every shard reads the same features |
| Partial outputs | Each shard publishes its partial output per position; after a spin barrier every shard sums them in the same order, so all shards see bit-identical predictions and the result does not depend on the thread count |
| Segment shuffles | Records are shuffled once globally at load time; each epoch then shuffles per-thread segments in parallel instead of one single-threaded pass over the whole file |

## Train commands

- `./target/release/arabica datagen <positions> [threads] [nodes] [output]` - generates training positions from self-play (all cores, 5000 nodes per move by default) and appends them to `<output>` (default `data/arabica-<version>-n<nodes>.bin`) in bullet's `ChessBoard` format; rerunning with the same settings resumes the same file
- `./target/release/arabica-train train <data> <checkpoint> [epochs] [threads]` - trains the network on a `datagen` file (40 epochs, all cores by default), resumes an existing checkpoint and exports a `.nnue` network next to it after every epoch; the whole file is loaded into memory (32 bytes per position)
- `./target/release/arabica-train export <checkpoint> [network]` - quantizes a checkpoint into the engine's `.nnue` format (default: the checkpoint path with the `.nnue` extension)
