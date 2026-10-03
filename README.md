# Arabica Chess

A chess engine written in Rust, built on bitboards.

UCI, Magic bitboards, Negamax, Iterative deepening, Principal variation search, Quiescence search, Check extension, Zobrist hashing, Repetition detection, Transposition table, Killer moves, History heuristic, Continuation history, Capture history, Null move pruning, Late move reductions, Reverse futility pruning, Futility pruning, Late move pruning, Static exchange evaluation, Tapered evaluation, Threat evaluation, Nonlinear king danger, Pawn structure evaluation, Endgame scaling, Pawn hash table, Texel tuning

## Use

`cargo build --release`
`./target/release/arabica`

**Commands**
- `uci` - name, author and uciok
- `isready` - readyok
- `setoption name Hash value <mb>` - resizes the transposition table (1-4096 MB, default 16)
- `setoption name Move Overhead value <ms>` - time reserved per move for GUI and network latency (0-5000 ms, default 50)
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
- `stop` - stops the search
- `quit` - stops the search and exits

**Tuning**
- `./target/release/arabica tune <dataset.epd> <params.rs> [epochs]` - Texel-tunes the evaluation weights on a labeled EPD dataset (for example Zurichess `quiet-labeled.epd`) and writes them in the `src/eval/params.rs` format
