# Arabica Chess

UCI chess engine in Rust powered by NNUE

## UCI commands

| Command | Description |
|---|---|
| `uci` | Name, author and uciok |
| `isready` | Readyok |
| `setoption name <name> value <value>` | Sets one of the UCI options listed below |
| `position startpos [moves e2e4 e7e5]` | Sets up the initial position |
| `position fen <fen> [moves g1f3 d7d5]` | Sets up a custom position |
| `d` | Prints the board, its FEN and Zobrist key |
| `ucinewgame` | Resets the board to the start position, clears the transposition table and move histories and reseeds the evaluation noise |
| `go` | Starts a search |
| `go depth 8` | Starts a search with depth 8 |
| `go nodes 5000` | Stops the search after 5000 nodes (the first iteration always completes) |
| `go infinite` | Searches until `stop` |
| `go movetime 1000` | Searches for 1000 ms |
| `go wtime <ms> btime <ms> [winc <ms>] [binc <ms>] [movestogo <n>]` | Searches with clock-based time management |
| `go perft <depth>` | Perft divide and the total node count |
| `bench [depth]` | Searches the Stockfish bench positions to a fixed depth (default 13) and prints total nodes and nps |
| `go ponder ...` | Searches on the opponent's time with the clock paused and holds `bestmove` until `ponderhit` or `stop` |
| `ponderhit` | The expected reply was played: the clock starts now and the ongoing search continues under the usual time limits |
| `stop` | Stops the search |
| `quit` | Stops the search and exits |

## UCI options

| Option | Type | Default | Range | Description |
|---|---|---|---|---|
| `Hash` | spin | 16 | 1-4096 | Transposition table size in MB; setting it reallocates and clears the table |
| `Move Overhead` | spin | 50 | 0-5000 | Time in ms reserved per move for GUI and network latency |
| `Ponder` | check | false | | Appends the expected reply to `bestmove` as `ponder <move>` |
| `UCI_LimitStrength` | check | false | | Plays at the strength set by `UCI_Elo` |
| `UCI_Elo` | spin | 1500 | 600-3000 | Target strength when `UCI_LimitStrength` is on |
| `EvalFile` | string | `<embedded>` | | Loads a `.nnue` network from the given path and clears the transposition table; `<embedded>` restores the built-in network, a file that fails to load keeps the current one |

Board representation and move generation are summarized in [docs/board.md](docs/board.md).

Search techniques are listed in [docs/search.md](docs/search.md).

Network details and the training pipeline are described in [docs/nnue.md](docs/nnue.md).
