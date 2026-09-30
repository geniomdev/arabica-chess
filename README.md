# Arabica Chess

A chess engine written in Rust, built on bitboards.

UCI, Magic bitboards, Negamax, Iterative deepening, Quiescence search, Check extension, Principal variation

## Use

`cargo build --release`
`./target/release/arabica`

**Commands**
`uci` - name, author and uciok
`isready` - readyok
`position startpos \| fen <fen> [moves ...]` - sets up the position
`position startpos moves e2e4 e7e5` - sets up the position
`d` - prints the board and its FEN
`ucinewgame` - resets the board to the start position
`go` - starts a search
`go depth 8` - starts a search with depth 8
`go perft <depth>` - perft divide and the total node count
`stop` - stops the search
`quit` - stops the search and exits

## Goals

+ Bitboards
+ FEN
+ Move generation
+ Make and unmake move
+ Search
+ UCI protocol
- Zobrist and repetition detection
- TT
