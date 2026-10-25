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

## Network history

| Network | Data | Training | Result |
|---|---|---|---|
| `net1` (0.2.0), `768 -> 128 x2 -> 1` | 100M positions from 964 632 self-play games of arabica 0.1.3 (HCE), 5000 nodes per move, 12 threads, 12 h | 40 epochs, ~25 s each; final loss 0.005618 train, 0.005643 validation | +219.9 ± 58.3 Elo vs 0.1.3 (HCE), 100 games, 10+0.1, `8moves_v3.pgn`: +61 -5 =34 |
| `net2` (0.3.0), `768 -> 512 x2 -> 1` | 200M positions from self-play of arabica 0.2.0 (`net1`), 5000 nodes per move, 12 threads, 21 h | 40 epochs, ~2:49 each, 1:52:38 total; final loss 0.006827 train, 0.006873 validation | +74.1 ± 46.6 Elo vs 0.2.0 (`net1`), 100 games, 10+0.1, `8moves_v3.pgn`: +34 -13 =53 |

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

## Training data

`datagen` stores whole games, and `arabica-train` decides which positions to train on when it loads them, so the filter can change without regenerating data.

| Part | Layout |
|---|---|
| File header | 8 bytes, `ARBGAME1` |
| Game | outcome for White (1 byte: 0 loss, 1 draw, 2 win), start FEN length (1 byte), start FEN, ply count (`u16`), then 4 bytes per ply |
| Ply | played move (`u16`, the engine's move encoding) and the search score of the position before it from the side to move (`i16`, centipawns or mate score) |

The start FEN is the position after the random opening. Every played move is the search's best move; a game adjudicated as decided ends before the ply that triggered the adjudication. A ply takes 4 bytes, about 5.5 bytes per position kept by the default filter, against 32 bytes in the position format.

| Filter (default) | Value |
|---|---|
| Skipped plies | first 6 plies after the random opening |
| Side to move in check | dropped |
| Best move a capture or a promotion | dropped |
| Score | \|score\| up to `max_score`, by default every non-mate score |

The default lives in `Filter::DEFAULT` (`src/nnue/games.rs`); `datagen` counts its target positions with it and `arabica-train` applies it when loading.

The trainer replays each game from its start position, checks every move for legality, and turns the kept positions into the same 32-byte records the position format stores, with the game outcome from the side to move. Loading runs in two passes over the file (count, then fill an exactly sized buffer), so memory peaks at the records themselves; 30M positions load in about 17 s on 12 threads. Files in the older position format (bullet's `ChessBoard`, 32 bytes per position, no header) still load as they are.

## Train commands

- `./target/release/arabica datagen <positions> [threads] [nodes] [output]` - plays self-play games (all cores, a soft limit of 2000 nodes per move by default: an iteration that reaches the limit is the last one, and a hard limit of 10x the soft one cuts runaway iterations; this averages about 5000 nodes per move) until `<positions>` positions pass the default filter and appends the games to `<output>` (default `data/arabica-<version>-n<nodes>.games`); rerunning with the same settings resumes the same file, and a file without the games header is refused
- `./target/release/arabica-train train <data> <checkpoint> [epochs] [threads]` - trains the network on a games file or a position file (40 epochs, all cores by default), resumes an existing checkpoint and exports a `.nnue` network next to it after every epoch; the kept positions are held in memory at 32 bytes each
- `./target/release/arabica-train export <checkpoint> [network]` - quantizes a checkpoint into the engine's `.nnue` format (default: the checkpoint path with the `.nnue` extension)
