use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::board::{Board, MAX_MOVES, MoveList};
use crate::eval::{evaluate, piece_value};
use crate::types::{Move, MoveKind, Piece};

pub const INFINITY: i32 = 32_000;
pub const MATE: i32 = 31_000;
pub const MAX_PLY: usize = 128;
pub const MAX_DEPTH: u8 = 64;

const MATE_BOUND: i32 = MATE - MAX_PLY as i32;
const TIME_CHECK_INTERVAL: u64 = 2048;
const PV_MOVE_SCORE: i32 = 1_000_000;
const CAPTURE_BASE_SCORE: i32 = 100_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchLimits {
    pub depth: Option<u8>,
    pub time: Option<Duration>,
    pub soft_time: Option<Duration>,
}

pub struct Iteration<'a> {
    pub depth: u8,
    pub score: i32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub pv: &'a [Move],
}

pub struct Searcher {
    limits: SearchLimits,
    stop_signal: Arc<AtomicBool>,
    start: Instant,
    nodes: u64,
    ply: usize,
    stopped: bool,
    pv: [[Move; MAX_PLY]; MAX_PLY],
    pv_len: [usize; MAX_PLY],
    previous_pv: [Move; MAX_PLY],
    previous_pv_len: usize,
}

struct MovePicker {
    moves: MoveList,
    scores: [i32; MAX_MOVES],
    next: usize,
}

impl MovePicker {
    fn new(board: &Board, moves: MoveList, pv_move: Option<Move>) -> Self {
        let mut scores = [0; MAX_MOVES];
        for (score, &candidate) in scores.iter_mut().zip(moves.as_slice()) {
            *score = if Some(candidate) == pv_move {
                PV_MOVE_SCORE
            } else {
                order_score(board, candidate)
            };
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

fn order_score(board: &Board, candidate: Move) -> i32 {
    let promotion = candidate.promotion().map_or(0, piece_value);
    if !candidate.is_capture() {
        return promotion;
    }
    let victim = match candidate.kind() {
        MoveKind::EnPassant => Piece::Pawn,
        _ => board.piece_on(candidate.to()).expect("piece on captured square"),
    };
    let attacker = board.piece_on(candidate.from()).expect("piece on origin square");
    CAPTURE_BASE_SCORE + promotion + 10 * piece_value(victim) - attacker_rank(attacker)
}

fn attacker_rank(attacker: Piece) -> i32 {
    match attacker {
        Piece::King => 1_000,
        other => piece_value(other),
    }
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
    let moves = if score > 0 { (plies + 1) / 2 } else { -plies / 2 };
    format!("mate {moves}")
}

impl Searcher {
    pub fn new(limits: SearchLimits, stop_signal: Arc<AtomicBool>) -> Self {
        Self {
            limits,
            stop_signal,
            start: Instant::now(),
            nodes: 0,
            ply: 0,
            stopped: false,
            pv: [[Move::NULL; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
            previous_pv: [Move::NULL; MAX_PLY],
            previous_pv_len: 0,
        }
    }

    pub fn search(
        &mut self,
        board: &mut Board,
        mut report: impl FnMut(&Iteration),
    ) -> (Move, i32) {
        self.start = Instant::now();
        self.nodes = 0;
        self.ply = 0;
        self.stopped = false;
        self.previous_pv_len = 0;

        let max_depth = self.limits.depth.unwrap_or(MAX_DEPTH).clamp(1, MAX_DEPTH);
        let mut best = (Move::NULL, 0);
        for depth in 1..=max_depth {
            let score = self.negamax(board, depth, -INFINITY, INFINITY);
            if self.stopped {
                break;
            }
            let pv = &self.pv[0][..self.pv_len[0]];
            best = (pv.first().copied().unwrap_or(Move::NULL), score);
            self.previous_pv[..pv.len()].copy_from_slice(pv);
            self.previous_pv_len = pv.len();
            report(&Iteration {
                depth,
                score,
                nodes: self.nodes,
                elapsed: self.start.elapsed(),
                pv,
            });
            if is_mate_score(score) || best.0 == Move::NULL || self.soft_time_expired() {
                break;
            }
        }
        if best.0 == Move::NULL {
            best.0 = first_legal_move(board).unwrap_or(Move::NULL);
        }
        best
    }

    fn soft_time_expired(&self) -> bool {
        self.limits
            .soft_time
            .is_some_and(|budget| self.start.elapsed() >= budget)
    }

    fn hard_limit_reached(&self) -> bool {
        self.stop_signal.load(Ordering::Relaxed)
            || self
                .limits
                .time
                .is_some_and(|budget| self.start.elapsed() >= budget)
    }

    fn pv_move(&self) -> Option<Move> {
        (self.ply < self.previous_pv_len).then(|| self.previous_pv[self.ply])
    }

    fn visit_node(&mut self) {
        self.nodes += 1;
        if self.nodes.is_multiple_of(TIME_CHECK_INTERVAL) && self.hard_limit_reached() {
            self.stopped = true;
        }
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

    fn is_in_check(board: &Board) -> bool {
        let us = board.state.active_color;
        board.is_attacked(board.king_square(us), us.opponent(), board.occupancy())
    }

    fn negamax(&mut self, board: &mut Board, depth: u8, mut alpha: i32, beta: i32) -> i32 {
        self.pv_len[self.ply] = self.ply;
        if self.ply > 0 && board.state.halfmove_clock >= 100 {
            return 0;
        }
        if self.ply >= MAX_PLY - 1 {
            return evaluate(board);
        }
        let in_check = Self::is_in_check(board);
        let depth = if in_check { depth + 1 } else { depth };
        if depth == 0 {
            return self.quiescence(board, alpha, beta);
        }
        self.visit_node();
        if self.stopped {
            return 0;
        }

        let mut moves = MoveList::new();
        board.generate_pseudo_legal(&mut moves);
        let picker = MovePicker::new(board, moves, self.pv_move());

        let mut best_score = -INFINITY;
        let mut legal_moves = 0;
        for candidate in picker {
            if !board.make_move(candidate) {
                continue;
            }
            legal_moves += 1;
            self.ply += 1;
            let score = -self.negamax(board, depth - 1, -beta, -alpha);
            self.ply -= 1;
            board.unmake_move();
            if self.stopped {
                return 0;
            }
            best_score = best_score.max(score);
            if score > alpha {
                alpha = score;
                self.update_pv(candidate);
                if alpha >= beta {
                    break;
                }
            }
        }

        match legal_moves {
            0 if in_check => -MATE + self.ply as i32,
            0 => 0,
            _ => best_score,
        }
    }

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32) -> i32 {
        self.visit_node();
        if self.stopped {
            return 0;
        }
        let stand_pat = evaluate(board);
        if self.ply >= MAX_PLY - 1 || stand_pat >= beta {
            return stand_pat;
        }
        alpha = alpha.max(stand_pat);

        let mut moves = MoveList::new();
        board.generate_captures(&mut moves);
        let picker = MovePicker::new(board, moves, None);

        let mut best_score = stand_pat;
        for candidate in picker {
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

    #[test]
    fn finds_best_move() {
        let cases = [
            ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 3, Some("a1a8"), Some(MATE - 1)),
            ("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", 3, Some("a8a1"), Some(MATE - 1)),
            ("4k3/8/8/8/8/8/8/RR4K1 w - - 0 1", 4, None, Some(MATE - 3)),
            ("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", 3, Some("d1d5"), None),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3, Some("0000"), Some(0)),
            ("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1", 3, Some("0000"), Some(-MATE)),
        ];

        for (fen, depth, expected_move, expected_score) in cases {
            let mut board: Board = fen
                .parse()
                .unwrap_or_else(|error| panic!("fen {fen:?} rejected: {error}"));
            let limits = SearchLimits {
                depth: Some(depth),
                ..SearchLimits::default()
            };
            let (best_move, score) =
                Searcher::new(limits, Arc::default()).search(&mut board, |_| {});
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
