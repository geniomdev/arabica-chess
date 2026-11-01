use std::sync::atomic::{AtomicBool, Ordering};

use crate::board::{Board, MoveList};
use crate::nnue;
use crate::search::{INFINITY, MATE};
use crate::types::Move;

pub const MAX_PLIES: u8 = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Search,
    Plies(u8),
}

pub struct MinimaxResult {
    pub best_move: Move,
    pub score: i32,
    pub nodes: u64,
}

pub struct Minimax<'a> {
    stop_signal: &'a AtomicBool,
    nodes: u64,
}

impl<'a> Minimax<'a> {
    pub fn new(stop_signal: &'a AtomicBool) -> Self {
        Self {
            stop_signal,
            nodes: 0,
        }
    }

    pub fn search(&mut self, board: &mut Board, plies: u8) -> MinimaxResult {
        let mut moves = MoveList::new();
        board.generate_pseudo_legal(&mut moves);
        let mut best_move = Move::NULL;
        let mut best_score = -INFINITY;
        for &candidate in moves.as_slice() {
            if self.stop_signal.load(Ordering::Relaxed) && best_move != Move::NULL {
                break;
            }
            if !board.make_move(candidate) {
                continue;
            }
            let score = -self.negamax(board, plies.saturating_sub(1), 1);
            board.unmake_move();
            if score > best_score {
                best_move = candidate;
                best_score = score;
            }
        }
        if best_move == Move::NULL {
            best_score = if board.in_check() { -MATE } else { 0 };
        }
        MinimaxResult {
            best_move,
            score: best_score,
            nodes: self.nodes,
        }
    }

    fn negamax(&mut self, board: &mut Board, plies: u8, ply: i32) -> i32 {
        self.nodes += 1;
        let mut moves = MoveList::new();
        board.generate_pseudo_legal(&mut moves);
        let mut best = -INFINITY;
        let mut has_legal_move = false;
        for &candidate in moves.as_slice() {
            if !board.make_move(candidate) {
                continue;
            }
            has_legal_move = true;
            if plies == 0 {
                board.unmake_move();
                break;
            }
            best = best.max(-self.negamax(board, plies - 1, ply + 1));
            board.unmake_move();
        }
        if !has_legal_move {
            return if board.in_check() { ply - MATE } else { 0 };
        }
        if plies == 0 {
            return nnue::evaluate(board);
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::testing::{board, find_move};
    use crate::board::{KIWIPETE, START_POSITION};
    use crate::search::is_mate_score;

    #[test]
    fn searches_every_position_to_the_given_depth() {
        let back_rank_mate = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1";
        let ladder_mate = "7k/8/8/8/8/8/R7/1R4K1 w - - 0 1";
        let stalemated = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1";
        let cases = [
            (START_POSITION, 1, None, None, Some(20)),
            (START_POSITION, 2, None, None, Some(20 + 400)),
            (START_POSITION, 3, None, None, Some(20 + 400 + 8_902)),
            (KIWIPETE, 2, None, None, Some(48 + 2_039)),
            (back_rank_mate, 1, Some("a1a8"), Some(MATE - 1), None),
            (back_rank_mate, 2, Some("a1a8"), Some(MATE - 1), None),
            (ladder_mate, 2, None, None, None),
            (ladder_mate, 3, None, Some(MATE - 3), None),
            (stalemated, 1, None, Some(0), Some(0)),
        ];

        for (fen, plies, expected_move, expected_score, expected_nodes) in cases {
            let mut position = board(fen);
            let stop_signal = AtomicBool::new(false);
            let result = Minimax::new(&stop_signal).search(&mut position, plies);
            let context = format!("fen {fen:?}, plies {plies}");
            if let Some(notation) = expected_move {
                assert_eq!(
                    result.best_move,
                    find_move(&position, notation),
                    "{context}"
                );
            }
            match expected_score {
                Some(score) => assert_eq!(result.score, score, "{context}"),
                None => assert!(!is_mate_score(result.score), "{context}"),
            }
            if let Some(nodes) = expected_nodes {
                assert_eq!(result.nodes, nodes, "{context}");
            }
            assert_eq!(position.to_fen(), board(fen).to_fen(), "{context}");
        }
    }
}
