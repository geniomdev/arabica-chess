# Arabica Chess

<p align="left">
  <a href="https://github.com/geniomdev/arabica-chess/releases/latest"><img src="https://img.shields.io/github/v/release/geniomdev/arabica-chess?logo=github&label=Release&color=green" alt="Release"></a>
  <a href="https://lichess.org/@/ArabicaNNUE/perf/bullet"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FArabicaNNUE&query=%24.perfs.bullet.rating&label=Bullet&logo=lichess&logoColor=white&color=blue" alt="Bullet"></a>
  <a href="https://lichess.org/@/ArabicaNNUE/perf/blitz"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FArabicaNNUE&query=%24.perfs.blitz.rating&label=Blitz&logo=lichess&logoColor=white&color=blue" alt="Blitz"></a>
  <a href="https://lichess.org/@/ArabicaNNUE/perf/rapid"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FArabicaNNUE&query=%24.perfs.rapid.rating&label=Rapid&logo=lichess&logoColor=white&color=blue" alt="Rapid"></a>
</p>

Arabica is a UCI chess engine written from scratch in Rust, with zero dependencies. The engine plays games against itself to produce training data, a built-in trainer learns from those games, and the resulting NNUE is compiled straight into the engine.

## UCI commands

| Command | Description |
|---|---|
| `uci` | Name, author and uciok |
| `isready` | Readyok |
| `setoption name <name> value <value>` | Sets one of the UCI options listed below |
| `position startpos [moves e2e4 e7e5]` | Sets up the initial position |
| `position fen <fen> [moves g1f3 d7d5]` | Sets up a custom position |
| `d` | Prints the board, its FEN and Zobrist key |
| `eval` | Prints the static NNUE evaluation of the current position in centipawns from the side to move's point of view |
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
| `Threads` | spin | 1 | 1-256 | Number of search threads (Lazy SMP: all threads search the same position and share the transposition table) |
| `Move Overhead` | spin | 50 | 0-5000 | Time in ms reserved per move for GUI and network latency |
| `Ponder` | check | false | | Appends the expected reply to `bestmove` as `ponder <move>` |
| `UCI_LimitStrength` | check | false | | Plays at the strength set by `UCI_Elo` |
| `UCI_Elo` | spin | 1500 | 600-3000 | Target strength when `UCI_LimitStrength` is on |
| `EvalFile` | string | `<embedded>` | | Loads a `.nnue` network from the given path and clears the transposition table; `<embedded>` restores the built-in network, a file that fails to load keeps the current one |
| `SyzygyPath` | string | `<empty>` | | Directories with Syzygy tablebase files, separated by `:` (`;` on Windows); `<empty>` turns tablebases off |

## Release builds

| Build | CPUs |
|---|---|
| `x86-64-v2` | Any x86-64 CPU with SSE4.2 and POPCNT (about 2009 and later); the output layer still switches to AVX2 or AVX-512 at run time when the CPU has them |
| `x86-64-v3` | AVX2, BMI2 and FMA (Intel Haswell, AMD Zen and later); the build for most machines |
| `x86-64-v4` | AVX-512 (AMD Zen 4, Intel Skylake-X, Ice Lake, Sapphire Rapids and later); stops with an illegal instruction error on other CPUs |
| `macos-arm64` | Apple silicon |

Board representation and move generation are summarized in [docs/board.md](docs/board.md).

Search techniques are listed in [docs/search.md](docs/search.md).

Network details and the training pipeline are described in [docs/nnue.md](docs/nnue.md).
