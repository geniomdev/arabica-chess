# Training

Evaluation uses a SCReLU NNUE with a lazily updated accumulator stack and an AVX2 output layer, trained on self-play data with its own CPU trainer.

## Current network

| Property | Value |
|---|---|
| File | `networks/net1.nnue`, embedded into the binary at build time; the `EvalFile` UCI option swaps in another `.nnue` at runtime (e.g. `option.EvalFile=data/nets/net2.nnue` in fastchess) without rebuilding |
| Architecture | `768 -> 128 x2 -> 1`, perspective network |
| Inputs | 768 = 2 sides (own / opponent) x 6 piece types x 64 squares, squares flipped vertically for Black |
| Hidden layer | 128 neurons per perspective, shared feature weights, accumulator updated incrementally on a lazy stack |
| Activation | SCReLU (`clamp(x, 0, QA)^2`) |
| Output | Single neuron over the concatenated side-to-move and opponent accumulators (256 inputs) |
| Parameters | 98 689 (`i16`, little-endian), 197 378 bytes |
| Quantization | QA = 255 (feature weights and biases), QB = 64 (output weights), output bias scaled by QA * QB |
| Evaluation scale | 400 (centipawns = output * 400 / (QA * QB)) |
| Inference | AVX2 `madd` output layer when available and the output weights fit, portable fallback otherwise |
| Special cases | Positions without enough material to mate evaluate to 0 |

## Train commands

- `./target/release/arabica datagen <positions> [threads] [nodes] [output]` - generates training positions from self-play (all cores, 5000 nodes per move by default) and appends them to `<output>` (default `data/arabica-<version>-n<nodes>.bin`) in bullet's `ChessBoard` format; rerunning with the same settings resumes the same file
- `./target/release/arabica train <data> <checkpoint> [epochs] [threads]` - trains the network on a `datagen` file (40 epochs, all cores by default), resumes an existing checkpoint and exports a `.nnue` network next to it after every epoch
- `./target/release/arabica export <checkpoint> [network]` - quantizes a checkpoint into the engine's `.nnue` format (default: the checkpoint path with the `.nnue` extension)
