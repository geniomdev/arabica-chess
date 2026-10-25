# Board

| Component | Implementation |
|---|---|
| Squares | `A1 = 0` ... `H8 = 63`, rank-major |
| Position | 6 piece bitboards + 2 color bitboards + a mailbox for O(1) piece lookup |
| Sliding attacks | Fancy magic bitboards with hardcoded factors, one shared table (107 648 entries) built at compile time |
| Leaper attacks | Knight, king and pawn tables built at compile time |
| Move | `u16`: from (6 bits), to (6 bits), kind (4 bits: quiet, double push, castles, capture, en passant, promotions) |
| Move generation | Pseudo-legal, in two stages: noisy (captures, promotions) and quiet (including castling) |
| Legality | `make_move` plays the move and undoes it if the own king is left in check, returning `false` |
| Unmake | Previous `GameState` is popped from the history stack, pieces are moved back |
| Zobrist | Keys from a fixed-seed SplitMix64, updated incrementally; en passant is hashed only when a pawn can actually capture |
| Repetitions | Scans reversible plies back by two; a repeat inside the search tree or a third occurrence counts as a draw |
| SEE | Swap algorithm with x-rays through the updated occupancy, answers `see(move) >= threshold` |
| NNUE | Every piece change is also pushed onto the accumulator stack |

Correctness is checked by perft against reference counts (start position, Kiwipete and others) in `cargo test`, and by `go perft <depth>` from UCI.
