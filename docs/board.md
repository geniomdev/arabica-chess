# Board

| Component | Implementation |
|---|---|
| Squares | `A1 = 0` ... `H8 = 63`, rank-major |
| Position | 6 piece bitboards + 2 color bitboards + a mailbox for O(1) piece lookup |
| Sliding attacks | Fancy magic bitboards with hardcoded factors, one shared table (107 648 entries) built at compile time |
| Leaper attacks | Knight, king and pawn tables built at compile time |
| Move | `u16`: from (6 bits), to (6 bits), kind (4 bits: quiet, double push, castles, capture, en passant, promotions); castling is stored as king-takes-own-rook, so `to` is the rook square |
| Castling | Chess960-aware: the rook squares come from the FEN and stay fixed for the game; `KingCastle` puts the king on g and the rook on f, `QueenCastle` on c and d |
| Castling FEN | Reads `KQkq`, X-FEN and Shredder-FEN (`HAha`); writes X-FEN, which is plain `KQkq` unless another rook stands further out on that wing |
| Move generation | Pseudo-legal, in two stages: noisy (captures, promotions) and quiet (including castling) |
| Castling generation | All squares the king and the rook cross must be empty except the two of them, and the king's path must not be attacked; a check revealed by the rook leaving its square is caught by the legality test in `make_move` |
| Legality | `make_move` plays the move and undoes it if the own king is left in check, returning `false` |
| Unmake | Previous `GameState` is popped from the history stack, pieces are moved back |
| Zobrist | Keys from a fixed-seed SplitMix64, updated incrementally; en passant is hashed only when a pawn can actually capture |
| History indexing | Quiet histories index castling by the king's landing square (g or c), so classical games search exactly as before the Chess960 encoding |
| Repetitions | Scans reversible plies back by two; a repeat inside the search tree or a third occurrence counts as a draw |
| SEE | Swap algorithm with x-rays through the updated occupancy, answers `see(move) >= threshold` |
| NNUE | Every piece change is also pushed onto the accumulator stack |

Correctness is checked by perft against reference counts (start position, Kiwipete, Chess960 positions and others, the Chess960 ones taken from Stockfish) in `cargo test`, and by `go perft <depth>` from UCI.
