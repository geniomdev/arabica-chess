# Arabica Chess

A chess engine written in Rust, built on bitboards.

UCI, Magic bitboards, Negamax, Iterative deepening, Principal variation search, Quiescence search, Check extension, Zobrist hashing, Repetition detection, Transposition table, Staged move generation, Killer moves, History heuristic, Continuation history, Capture history, Null move pruning, Late move reductions, Reverse futility pruning, Futility pruning, Late move pruning, Static exchange evaluation, Tapered evaluation, Threat evaluation, Nonlinear king danger, Pawn structure evaluation, Endgame scaling, Pawn hash table, Texel tuning, NNUE evaluation (768 -> 128 x2 -> 1 SCReLU, lazily updated accumulator stack, AVX2 output layer, self-play data, own CPU trainer)

## Use

`cargo build --release`
`./target/release/arabica`

**Commands**
- `uci` - name, author and uciok
- `isready` - readyok
- `setoption name Hash value <mb>` - resizes the transposition table (1-4096 MB, default 16)
- `setoption name Move Overhead value <ms>` - time reserved per move for GUI and network latency (0-5000 ms, default 50)
- `setoption name Use NNUE value <true|false>` - evaluates positions with the embedded network `networks/net1.nnue` (default true) or with the classical hand-crafted evaluation; positions without enough material to mate always evaluate to 0 under the network
- `setoption name Ponder value <true|false>` - appends the expected reply to `bestmove` as `ponder <move>` (default false)
- `setoption name UCI_LimitStrength value <true|false>` - plays at the strength set by `UCI_Elo` (default false)
- `setoption name UCI_Elo value <elo>` - target strength when `UCI_LimitStrength` is on (600-3000, default 1500); weakened by a per-move node limit and deterministic Zobrist-keyed noise in the static evaluation, both interpolated from Elo anchors in `src/strength.rs`
- `position startpos \| fen <fen> [moves ...]` - sets up the position
- `position startpos moves e2e4 e7e5` - sets up the position
- `d` - prints the board, its FEN and Zobrist key
- `ucinewgame` - resets the board to the start position, clears the transposition table and move histories and reseeds the evaluation noise
- `go` - starts a search
- `go depth 8` - starts a search with depth 8
- `go nodes 5000` - stops the search after 5000 nodes (the first iteration always completes)
- `go perft <depth>` - perft divide and the total node count
- `bench [depth]` - searches the Stockfish bench positions to a fixed depth (default 13) and prints total nodes and nps
- `go ponder ...` - searches on the opponent's time with the clock paused and holds `bestmove` until `ponderhit` or `stop`
- `ponderhit` - the expected reply was played: the clock starts now and the ongoing search continues under the usual time limits
- `stop` - stops the search
- `quit` - stops the search and exits

**Tuning**
- `./target/release/arabica tune <dataset.epd> <params.rs> [epochs]` - Texel-tunes the evaluation weights on a labeled EPD dataset (for example Zurichess `quiet-labeled.epd`) and writes them in the `src/eval/params.rs` format
- `./target/release/arabica datagen <positions> [threads] [nodes] [output]` - appends NNUE training positions from self-play (all cores and 5000 nodes per move by default) to `<output>` (default `data/arabica-<version>-n<nodes>.bin`, so rerunning with the same settings resumes the same file; `/data` is git-ignored) in bullet's 32-byte `ChessBoard` format: games start with 8 random plies whose position is color-flipped half of the time so either side can move first, openings scored beyond 500 cp are dropped, wins are adjudicated after 4 plies at 2500 cp or more, and positions in check, with a capture or promotion as the best move or with a mate score are skipped
- `./target/release/arabica train <data> <checkpoint> [epochs] [threads]` - trains the `768 -> 128 x2 -> 1` SCReLU network in floating point on a `datagen` file (40 epochs and all cores by default): 1% of the positions (at most 1M) is held out for validation, batches of 16384 positions are fitted with Adam on the squared error between `sigmoid(output)` and `0.75 * sigmoid(score / 400) + 0.25 * result`, the learning rate follows a cosine from 0.001 to 0.00001, and weights are clipped to [-1.98, 1.98]; the checkpoint is rewritten after every epoch and an existing one is resumed; after every epoch the network is also exported next to the checkpoint with the `.nnue` extension, and the final summary reports how far the quantized evaluation drifts from the float one on validation positions
- `./target/release/arabica export <checkpoint> [network]` - quantizes a training checkpoint into the engine's network format (default: the checkpoint path with the `.nnue` extension): feature weights and biases are scaled by 255, output weights by 64 and the output bias by 255 * 64, rounded to `i16` and written little-endian in the order of `nnue::Network`
