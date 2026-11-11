# Search

- Negamax
- Iterative deepening
- Lazy SMP with best-thread voting
- Aspiration windows with fail-high depth reduction
- Principal variation search
- Expected cut-node tracking
- Quiescence search
- Singular extensions with multi-cut
- Repetition detection
- Fifty-move and insufficient material draws
- Transposition table
- Static eval in transposition table
- Killer moves
- History heuristic bucketed by threats on the from and to squares
- Threat escape bonus in quiet move ordering
- Continuation history
- Capture history
- Correction history
- Improving heuristic
- Null move pruning
- ProbCut
- Internal iterative reductions, deeper in expected cut-nodes
- Late move reductions, shallower for threat escapes
- Reverse futility pruning
- Futility pruning
- Late move pruning
- History pruning
- Static exchange evaluation
- SEE pruning
- Quiescence futility pruning
- Time management
- Syzygy tablebases (WDL and DTZ)

## Multithreaded search

| Threads | 2 | 4 | 8 |
|---|---|---|---|
| Elo vs 1 thread | +85 ± 23 | +158 ±35 | +214 ±35 |
