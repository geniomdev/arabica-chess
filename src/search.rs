use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use crate::board::{Board, MAX_MOVES, MoveList, see_value};
use crate::eval::evaluate;
use crate::strength::EvalNoise;
use crate::tt::{Bound, Store, TranspositionTable};
use crate::types::{Color, Move, MoveKind, Piece, SQUARES};

pub const INFINITY: i32 = 32_000;
pub const MATE: i32 = 31_000;
pub const MAX_PLY: usize = 128;
pub const MAX_DEPTH: u8 = 64;

pub const MATE_BOUND: i32 = MATE - MAX_PLY as i32;
const TIME_CHECK_INTERVAL: u64 = 2048;

const HASH_MOVE_SCORE: i32 = 1_000_000;
const GOOD_NOISY_SCORE: i32 = 200_000;
const KILLER_SCORES: [i32; 2] = [100_000, 99_000];
const BAD_NOISY_SCORE: i32 = -200_000;

const MAX_HISTORY: i32 = 16_384;
const MAX_HISTORY_BONUS: i32 = 1_200;
const TRACKED_QUIETS: usize = 64;

const REVERSE_FUTILITY_DEPTH: i32 = 8;
const REVERSE_FUTILITY_MARGIN: i32 = 80;
const NULL_MOVE_DEPTH: i32 = 3;
const LATE_MOVE_PRUNING_DEPTH: i32 = 8;
const FUTILITY_DEPTH: i32 = 6;
const FUTILITY_BASE: i32 = 100;
const FUTILITY_MARGIN: i32 = 100;
const NOISY_SEE_DEPTH: i32 = 6;
const NOISY_SEE_MARGIN: i32 = 100;
const REDUCTION_DEPTH: i32 = 3;
const REDUCTION_HISTORY_DIVISOR: i32 = 8_192;
const REDUCTION_TABLE_SIZE: usize = 64;

const STABILITY_TIME_SCALES: [f64; 5] = [2.0, 1.4, 1.1, 0.9, 0.8];
const SCORE_DROP_RANGE: (i32, i32) = (-50, 100);
const SCORE_DROP_TIME_DIVISOR: f64 = 200.0;
const BEST_MOVE_NODES_PIVOT: f64 = 1.5;
const BEST_MOVE_NODES_SCALE: f64 = 1.35;
const SOFT_TIME_SCALE_RANGE: (f64, f64) = (0.5, 2.5);

type HistoryTable = [[[i32; SQUARES]; SQUARES]; 2];
type RootMoveNodes = [[u64; SQUARES]; SQUARES];

static REDUCTIONS: LazyLock<[[i32; REDUCTION_TABLE_SIZE]; REDUCTION_TABLE_SIZE]> =
    LazyLock::new(|| {
        let mut table = [[0; REDUCTION_TABLE_SIZE]; REDUCTION_TABLE_SIZE];
        for (depth, row) in table.iter_mut().enumerate().skip(1) {
            for (move_number, reduction) in row.iter_mut().enumerate().skip(1) {
                let scaled = (depth as f64).ln() * (move_number as f64).ln() / 2.25;
                *reduction = (0.75 + scaled) as i32;
            }
        }
        table
    });

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchLimits {
    pub depth: Option<u8>,
    pub time: Option<Duration>,
    pub soft_time: Option<Duration>,
    pub nodes: Option<u64>,
}

pub struct Iteration<'a> {
    pub depth: u8,
    pub score: i32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub pv: &'a [Move],
    pub hashfull: usize,
}

pub struct Searcher {
    limits: SearchLimits,
    table: TranspositionTable,
    stop_signal: Arc<AtomicBool>,
    start: Instant,
    nodes: u64,
    ply: usize,
    stopped: bool,
    first_iteration_completed: bool,
    eval_noise: EvalNoise,
    pv: [[Move; MAX_PLY]; MAX_PLY],
    pv_len: [usize; MAX_PLY],
    killers: [[Move; 2]; MAX_PLY],
    history: Box<HistoryTable>,
    root_move_nodes: Box<RootMoveNodes>,
}

struct MovePicker {
    moves: MoveList,
    scores: [i32; MAX_MOVES],
    next: usize,
}

impl MovePicker {
    fn new(
        board: &Board,
        moves: MoveList,
        hash_move: Option<Move>,
        killers: [Move; 2],
        history: &HistoryTable,
    ) -> Self {
        Self::scored(moves, |candidate| {
            if Some(candidate) == hash_move {
                return HASH_MOVE_SCORE;
            }
            if is_noisy(candidate) {
                let base = if board.see(candidate, 0) {
                    GOOD_NOISY_SCORE
                } else {
                    BAD_NOISY_SCORE
                };
                return base + noisy_score(board, candidate);
            }
            match killers.iter().position(|&killer| killer == candidate) {
                Some(rank) => KILLER_SCORES[rank],
                None => history_score(history, board.state.active_color, candidate),
            }
        })
    }

    fn noisy(board: &Board, moves: MoveList) -> Self {
        Self::scored(moves, |candidate| noisy_score(board, candidate))
    }

    fn scored(moves: MoveList, score: impl Fn(Move) -> i32) -> Self {
        let mut scores = [0; MAX_MOVES];
        for (slot, &candidate) in scores.iter_mut().zip(moves.as_slice()) {
            *slot = score(candidate);
        }
        Self {
            moves,
            scores,
            next: 0,
        }
    }
}

impl Iterator for MovePicker {
    type Item = Move;

    fn next(&mut self) -> Option<Move> {
        let moves = self.moves.as_mut_slice();
        let remaining = self.next..moves.len();
        let best = remaining.max_by_key(|&index| self.scores[index])?;
        moves.swap(self.next, best);
        self.scores.swap(self.next, best);
        self.next += 1;
        Some(moves[self.next - 1])
    }
}

fn is_noisy(candidate: Move) -> bool {
    candidate.is_capture() || candidate.is_promotion()
}

fn noisy_score(board: &Board, candidate: Move) -> i32 {
    let promotion = candidate.promotion().map_or(0, see_value);
    if !candidate.is_capture() {
        return promotion;
    }
    let victim = match candidate.kind() {
        MoveKind::EnPassant => Piece::Pawn,
        _ => board
            .piece_on(candidate.to())
            .expect("piece on captured square"),
    };
    let attacker = board
        .piece_on(candidate.from())
        .expect("piece on origin square");
    promotion + 10 * see_value(victim) - attacker_rank(attacker)
}

fn attacker_rank(attacker: Piece) -> i32 {
    match attacker {
        Piece::King => 1_000,
        other => see_value(other),
    }
}

fn history_score(history: &HistoryTable, side: Color, candidate: Move) -> i32 {
    history[side.index()][candidate.from().index()][candidate.to().index()]
}

fn update_history(entry: &mut i32, bonus: i32) {
    *entry += bonus - *entry * bonus.abs() / MAX_HISTORY;
}

fn soft_time_scale(stable_iterations: usize, score_drop: i32, best_move_node_share: f64) -> f64 {
    let stability = STABILITY_TIME_SCALES[stable_iterations.min(STABILITY_TIME_SCALES.len() - 1)];
    let (smallest_drop, largest_drop) = SCORE_DROP_RANGE;
    let score_trend =
        1.0 + f64::from(score_drop.clamp(smallest_drop, largest_drop)) / SCORE_DROP_TIME_DIVISOR;
    let node_share = (BEST_MOVE_NODES_PIVOT - best_move_node_share) * BEST_MOVE_NODES_SCALE;
    let (lowest, highest) = SOFT_TIME_SCALE_RANGE;
    (stability * score_trend * node_share).clamp(lowest, highest)
}

fn has_non_pawn_material(board: &Board, side: Color) -> bool {
    let pawns_and_kings = board.pieces(Piece::Pawn) | board.pieces(Piece::King);
    !(board.occupied_by(side) & !pawns_and_kings).is_empty()
}

fn late_move_reduction(depth: i32, move_number: usize) -> i32 {
    let depth = (depth as usize).min(REDUCTION_TABLE_SIZE - 1);
    REDUCTIONS[depth][move_number.min(REDUCTION_TABLE_SIZE - 1)]
}

fn first_legal_move(board: &mut Board) -> Option<Move> {
    let mut moves = MoveList::new();
    board.generate_pseudo_legal(&mut moves);
    moves.as_slice().iter().copied().find(|&candidate| {
        let legal = board.make_move(candidate);
        if legal {
            board.unmake_move();
        }
        legal
    })
}

pub fn is_mate_score(score: i32) -> bool {
    score.abs() >= MATE_BOUND
}

pub fn uci_score(score: i32) -> String {
    if !is_mate_score(score) {
        return format!("cp {score}");
    }
    let plies = MATE - score.abs();
    let moves = if score > 0 {
        (plies + 1) / 2
    } else {
        -plies / 2
    };
    format!("mate {moves}")
}

impl Searcher {
    pub fn new(
        limits: SearchLimits,
        stop_signal: Arc<AtomicBool>,
        table: TranspositionTable,
    ) -> Self {
        Self {
            limits,
            table,
            stop_signal,
            start: Instant::now(),
            nodes: 0,
            ply: 0,
            stopped: false,
            first_iteration_completed: false,
            eval_noise: EvalNoise::default(),
            pv: [[Move::NULL; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
            killers: [[Move::NULL; 2]; MAX_PLY],
            history: Box::new([[[0; SQUARES]; SQUARES]; 2]),
            root_move_nodes: Box::new([[0; SQUARES]; SQUARES]),
        }
    }

    pub fn with_eval_noise(mut self, eval_noise: EvalNoise) -> Self {
        self.eval_noise = eval_noise;
        self
    }

    pub fn search(&mut self, board: &mut Board, mut report: impl FnMut(&Iteration)) -> (Move, i32) {
        self.start = Instant::now();
        self.nodes = 0;
        self.ply = 0;
        self.stopped = false;
        self.first_iteration_completed = false;
        self.root_move_nodes.fill([0; SQUARES]);

        let max_depth = self.limits.depth.unwrap_or(MAX_DEPTH).clamp(1, MAX_DEPTH);
        let mut best = (Move::NULL, 0);
        let mut stable_iterations = 0;
        for depth in 1..=max_depth {
            let score = self.negamax(board, i32::from(depth), -INFINITY, INFINITY);
            let root_move_searched = self.pv_len[0] > 0;
            if self.stopped && !root_move_searched {
                break;
            }
            let pv = &self.pv[0][..self.pv_len[0]];
            let previous = best;
            best = (pv.first().copied().unwrap_or(Move::NULL), score);
            stable_iterations = if best.0 == previous.0 {
                stable_iterations + 1
            } else {
                0
            };
            report(&Iteration {
                depth,
                score,
                nodes: self.nodes,
                elapsed: self.start.elapsed(),
                pv,
                hashfull: self.table.hashfull(),
            });
            let score_drop = if depth == 1 { 0 } else { previous.1 - score };
            let time_scale = soft_time_scale(
                stable_iterations,
                score_drop,
                self.best_move_node_share(best.0),
            );
            self.first_iteration_completed = true;
            if self.stopped
                || self.node_limit_reached()
                || is_mate_score(score)
                || best.0 == Move::NULL
                || self.soft_time_expired(time_scale)
            {
                break;
            }
        }
        if best.0 == Move::NULL {
            best.0 = first_legal_move(board).unwrap_or(Move::NULL);
        }
        best
    }

    pub fn into_table(self) -> TranspositionTable {
        self.table
    }

    fn soft_time_expired(&self, scale: f64) -> bool {
        self.limits
            .soft_time
            .is_some_and(|budget| self.start.elapsed() >= budget.mul_f64(scale))
    }

    fn best_move_node_share(&self, best_move: Move) -> f64 {
        if self.nodes == 0 || best_move == Move::NULL {
            return 0.0;
        }
        let best_move_nodes =
            self.root_move_nodes[best_move.from().index()][best_move.to().index()];
        best_move_nodes as f64 / self.nodes as f64
    }

    fn hard_limit_reached(&self) -> bool {
        self.stop_signal.load(Ordering::Relaxed)
            || self
                .limits
                .time
                .is_some_and(|budget| self.start.elapsed() >= budget)
    }

    fn node_limit_reached(&self) -> bool {
        self.first_iteration_completed && self.limits.nodes.is_some_and(|limit| self.nodes >= limit)
    }

    fn visit_node(&mut self) {
        self.nodes += 1;
        if self.node_limit_reached()
            || (self.nodes.is_multiple_of(TIME_CHECK_INTERVAL) && self.hard_limit_reached())
        {
            self.stopped = true;
        }
    }

    fn static_evaluation(&self, board: &Board) -> i32 {
        evaluate(board) + self.eval_noise.offset(board.state.zobrist_key)
    }

    fn update_pv(&mut self, best: Move) {
        let ply = self.ply;
        let child_len = self.pv_len[ply + 1].max(ply + 1);
        self.pv[ply][ply] = best;
        for index in ply + 1..child_len {
            self.pv[ply][index] = self.pv[ply + 1][index];
        }
        self.pv_len[ply] = child_len;
    }

    fn reward_quiet(&mut self, side: Color, cutoff: Move, tried: &[Move], depth: i32) {
        let killers = &mut self.killers[self.ply];
        if killers[0] != cutoff {
            killers[1] = killers[0];
            killers[0] = cutoff;
        }
        let bonus = (16 * depth * depth).min(MAX_HISTORY_BONUS);
        let side_history = &mut self.history[side.index()];
        update_history(
            &mut side_history[cutoff.from().index()][cutoff.to().index()],
            bonus,
        );
        for &quiet in tried {
            update_history(
                &mut side_history[quiet.from().index()][quiet.to().index()],
                -bonus,
            );
        }
    }

    fn negamax(&mut self, board: &mut Board, depth: i32, mut alpha: i32, beta: i32) -> i32 {
        self.pv_len[self.ply] = self.ply;
        let is_root = self.ply == 0;
        let is_pv = beta - alpha > 1;
        if !is_root && (board.state.halfmove_clock >= 100 || board.is_repetition(self.ply)) {
            return 0;
        }
        if self.ply >= MAX_PLY - 1 {
            return self.static_evaluation(board);
        }
        let in_check = board.in_check();
        let depth = if in_check { depth + 1 } else { depth };
        if depth <= 0 {
            return self.quiescence(board, alpha, beta);
        }
        self.visit_node();
        if self.stopped {
            return 0;
        }
        self.killers[self.ply + 1] = [Move::NULL; 2];

        let key = board.state.zobrist_key;
        let hit = self.table.probe(key, self.ply);
        let table_depth = depth.min(i32::from(u8::MAX)) as u8;
        if let Some(score) = hit
            .filter(|_| !is_pv)
            .and_then(|hit| hit.cutoff_score(table_depth, alpha, beta))
        {
            return score;
        }
        let hash_move = hit.and_then(|hit| hit.best_move);

        let us = board.state.active_color;
        let static_eval = if in_check {
            -INFINITY
        } else {
            self.static_evaluation(board)
        };
        let prunable = !is_pv && !in_check;

        if prunable
            && depth <= REVERSE_FUTILITY_DEPTH
            && !is_mate_score(beta)
            && static_eval - REVERSE_FUTILITY_MARGIN * depth >= beta
        {
            return static_eval;
        }

        if prunable
            && depth >= NULL_MOVE_DEPTH
            && static_eval >= beta
            && board.state.played_move != Move::NULL
            && has_non_pawn_material(board, us)
        {
            let reduction = 3 + depth / 4 + ((static_eval - beta) / 200).min(3);
            board.make_null_move();
            self.ply += 1;
            let score = -self.negamax(board, depth - 1 - reduction, -beta, -beta + 1);
            self.ply -= 1;
            board.unmake_null_move();
            if self.stopped {
                return 0;
            }
            if score >= beta {
                return if is_mate_score(score) { beta } else { score };
            }
        }

        let mut moves = MoveList::new();
        board.generate_pseudo_legal(&mut moves);
        let picker = MovePicker::new(
            board,
            moves,
            hash_move,
            self.killers[self.ply],
            &self.history,
        );

        let original_alpha = alpha;
        let mut best_score = -INFINITY;
        let mut best_move = Move::NULL;
        let mut legal_moves = 0;
        let mut tried_quiets = [Move::NULL; TRACKED_QUIETS];
        let mut tried_quiet_count = 0;
        for candidate in picker {
            let is_quiet = !is_noisy(candidate);
            if !is_root && !in_check && best_score > -MATE_BOUND {
                if is_quiet {
                    if !is_pv
                        && depth <= LATE_MOVE_PRUNING_DEPTH
                        && legal_moves >= 3 + depth * depth
                    {
                        continue;
                    }
                    if depth <= FUTILITY_DEPTH
                        && static_eval + FUTILITY_BASE + FUTILITY_MARGIN * depth <= alpha
                    {
                        continue;
                    }
                } else if depth <= NOISY_SEE_DEPTH
                    && !board.see(candidate, -NOISY_SEE_MARGIN * depth)
                {
                    continue;
                }
            }
            let nodes_before = self.nodes;
            if !board.make_move(candidate) {
                continue;
            }
            legal_moves += 1;
            self.ply += 1;
            let new_depth = depth - 1;
            let score = if legal_moves == 1 {
                -self.negamax(board, new_depth, -beta, -alpha)
            } else {
                let gives_check = board.in_check();
                let mut reduction = 0;
                if depth >= REDUCTION_DEPTH && is_quiet && !in_check && !gives_check {
                    reduction = late_move_reduction(depth, legal_moves as usize);
                    reduction -= i32::from(is_pv);
                    reduction -= i32::from(self.killers[self.ply - 1].contains(&candidate));
                    reduction -=
                        history_score(&self.history, us, candidate) / REDUCTION_HISTORY_DIVISOR;
                    reduction = reduction.clamp(0, new_depth - 1);
                }
                let mut score = -self.negamax(board, new_depth - reduction, -alpha - 1, -alpha);
                if !self.stopped && score > alpha && reduction > 0 {
                    score = -self.negamax(board, new_depth, -alpha - 1, -alpha);
                }
                if !self.stopped && score > alpha && score < beta {
                    score = -self.negamax(board, new_depth, -beta, -alpha);
                }
                score
            };
            self.ply -= 1;
            board.unmake_move();
            if is_root {
                self.root_move_nodes[candidate.from().index()][candidate.to().index()] +=
                    self.nodes - nodes_before;
            }
            if self.stopped {
                return if is_root { best_score } else { 0 };
            }
            best_score = best_score.max(score);
            if score > alpha {
                alpha = score;
                best_move = candidate;
                self.update_pv(candidate);
                if alpha >= beta {
                    if is_quiet {
                        self.reward_quiet(us, candidate, &tried_quiets[..tried_quiet_count], depth);
                    }
                    break;
                }
            }
            if is_quiet && tried_quiet_count < TRACKED_QUIETS {
                tried_quiets[tried_quiet_count] = candidate;
                tried_quiet_count += 1;
            }
        }

        let score = match legal_moves {
            0 if in_check => -MATE + self.ply as i32,
            0 => 0,
            _ => best_score,
        };
        let bound = if legal_moves == 0 {
            Bound::Exact
        } else if score >= beta {
            Bound::Lower
        } else if score <= original_alpha {
            Bound::Upper
        } else {
            Bound::Exact
        };
        self.table.store(Store {
            key,
            best_move,
            score,
            depth: table_depth,
            bound,
            ply: self.ply,
        });
        score
    }

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32) -> i32 {
        self.visit_node();
        if self.stopped {
            return 0;
        }
        let stand_pat = self.static_evaluation(board);
        if self.ply >= MAX_PLY - 1 || stand_pat >= beta {
            return stand_pat;
        }
        alpha = alpha.max(stand_pat);

        let mut moves = MoveList::new();
        board.generate_captures(&mut moves);
        let picker = MovePicker::noisy(board, moves);

        let mut best_score = stand_pat;
        for candidate in picker {
            if !board.see(candidate, 0) {
                continue;
            }
            if !board.make_move(candidate) {
                continue;
            }
            self.ply += 1;
            let score = -self.quiescence(board, -beta, -alpha);
            self.ply -= 1;
            board.unmake_move();
            if self.stopped {
                return 0;
            }
            best_score = best_score.max(score);
            if score > alpha {
                alpha = score;
                if alpha >= beta {
                    break;
                }
            }
        }
        best_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::START_POSITION;
    use crate::board::testing::board;

    #[test]
    fn finds_best_move() {
        let cases = [
            (
                "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
                3,
                Some("a1a8"),
                Some(MATE - 1),
            ),
            (
                "r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1",
                3,
                Some("a8a1"),
                Some(MATE - 1),
            ),
            ("4k3/8/8/8/8/8/8/RR4K1 w - - 0 1", 4, None, Some(MATE - 3)),
            ("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", 3, Some("d1d5"), None),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3, Some("0000"), Some(0)),
            (
                "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1",
                3,
                Some("0000"),
                Some(-MATE),
            ),
        ];

        for (fen, depth, expected_move, expected_score) in cases {
            let mut board = board(fen);
            let limits = SearchLimits {
                depth: Some(depth),
                ..SearchLimits::default()
            };
            let (best_move, score) =
                Searcher::new(limits, Arc::default(), TranspositionTable::new(1))
                    .search(&mut board, |_| {});
            if let Some(expected) = expected_move {
                assert_eq!(best_move.to_string(), expected, "fen {fen:?}");
            }
            if let Some(expected) = expected_score {
                assert_eq!(score, expected, "fen {fen:?}");
            }
            assert_eq!(board.to_fen(), fen, "board restored for fen {fen:?}");
        }
    }

    #[test]
    fn keeps_best_move_from_interrupted_iteration() {
        let cases = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];

        for fen in cases {
            let mut board = board(fen);
            let limits = SearchLimits {
                time: Some(Duration::ZERO),
                ..SearchLimits::default()
            };
            let mut last_report = None;
            let (best_move, _) = Searcher::new(limits, Arc::default(), TranspositionTable::new(1))
                .search(&mut board, |iteration| {
                    last_report = Some((iteration.nodes, iteration.pv[0]))
                });
            let (nodes, pv_head) = last_report.expect("at least one iteration reported");
            assert_eq!(nodes, TIME_CHECK_INTERVAL, "fen {fen:?}");
            assert_eq!(best_move, pv_head, "fen {fen:?}");
            assert_eq!(board.to_fen(), fen, "board restored for fen {fen:?}");
        }
    }

    #[test]
    fn completes_first_iteration_under_node_limit() {
        let cases = [
            (START_POSITION, 1),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                1,
            ),
            (START_POSITION, 5_000),
        ];

        for (fen, node_limit) in cases {
            let mut board = board(fen);
            let limits = SearchLimits {
                nodes: Some(node_limit),
                ..SearchLimits::default()
            };
            let noise = EvalNoise {
                amplitude: 400,
                seed: 42,
            };
            let mut reports = Vec::new();
            let (best_move, _) = Searcher::new(limits, Arc::default(), TranspositionTable::new(1))
                .with_eval_noise(noise)
                .search(&mut board, |iteration| {
                    reports.push((iteration.depth, iteration.nodes, iteration.pv[0]))
                });
            let &(first_depth, first_nodes, _) = reports.first().expect("an iteration reported");
            let &(_, last_nodes, pv_head) = reports.last().expect("an iteration reported");
            assert_eq!(first_depth, 1, "fen {fen:?}");
            assert!(last_nodes <= node_limit.max(first_nodes), "fen {fen:?}");
            assert_ne!(best_move, Move::NULL, "fen {fen:?}");
            assert_eq!(best_move, pv_head, "fen {fen:?}");
            assert_eq!(board.to_fen(), fen, "board restored for fen {fen:?}");
        }
    }

    #[test]
    fn scales_soft_time_by_search_instability() {
        let cases = [
            (0, 0, 0.5, 2.5),
            (2, 100, 0.5, 1.1 * 1.5 * 1.35),
            (3, 40, 0.8, 0.9 * 1.2 * 0.7 * 1.35),
            (4, 0, 0.9, 0.8 * 0.6 * 1.35),
            (10, 0, 0.9, 0.8 * 0.6 * 1.35),
            (4, -100, 1.0, 0.5),
        ];

        for (stable_iterations, score_drop, node_share, expected) in cases {
            let scale = soft_time_scale(stable_iterations, score_drop, node_share);
            assert!(
                (scale - expected).abs() < 1e-9,
                "stable {stable_iterations}, drop {score_drop}, share {node_share}: {scale}"
            );
        }
    }

    #[test]
    fn formats_uci_scores() {
        let cases = [
            (0, "cp 0"),
            (-35, "cp -35"),
            (MATE - 1, "mate 1"),
            (MATE - 3, "mate 2"),
            (-MATE + 2, "mate -1"),
            (-MATE, "mate 0"),
        ];

        for (score, expected) in cases {
            assert_eq!(uci_score(score), expected, "score {score}");
        }
    }
}
