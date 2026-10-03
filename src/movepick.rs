use crate::board::{Board, MAX_MOVES, MoveList, see_value};
use crate::history::{Continuations, History, PieceTo};
use crate::types::{Move, MoveKind, Piece};

const CAPTURE_HISTORY_ORDER_DIVISOR: i32 = 8;
const KING_ATTACKER_RANK: i32 = 1_000;

pub fn is_noisy(candidate: Move) -> bool {
    candidate.is_capture() || candidate.is_promotion()
}

pub fn moved_piece(board: &Board, candidate: Move) -> PieceTo {
    PieceTo {
        side: board.state.active_color,
        piece: board
            .piece_on(candidate.from())
            .expect("piece on origin square"),
        to: candidate.to(),
    }
}

pub fn captured_piece(board: &Board, candidate: Move) -> Option<Piece> {
    match candidate.kind() {
        MoveKind::EnPassant => Some(Piece::Pawn),
        _ if candidate.is_capture() => Some(
            board
                .piece_on(candidate.to())
                .expect("piece on captured square"),
        ),
        _ => None,
    }
}

fn noisy_score(board: &Board, candidate: Move, history: &History) -> i32 {
    let promotion = candidate.promotion().map_or(0, see_value);
    let moved = moved_piece(board, candidate);
    let victim = captured_piece(board, candidate);
    let victim_attacker_score = victim.map_or(0, |victim| {
        10 * see_value(victim) - attacker_rank(moved.piece)
    });
    promotion
        + victim_attacker_score
        + history.capture_score(moved, victim) / CAPTURE_HISTORY_ORDER_DIVISOR
}

fn attacker_rank(attacker: Piece) -> i32 {
    match attacker {
        Piece::King => KING_ATTACKER_RANK,
        other => see_value(other),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    HashMove,
    GenerateNoisy,
    GoodNoisy,
    Killers,
    GenerateQuiets,
    Quiets,
    BadNoisy,
    Done,
}

pub struct MovePicker {
    stage: Stage,
    hash_move: Option<Move>,
    killers: [Move; 2],
    next_killer: usize,
    moves: MoveList,
    scores: [i32; MAX_MOVES],
    next: usize,
    bad_noisy_end: usize,
    next_bad_noisy: usize,
    quiets_skipped: bool,
    noisy_only: bool,
}

impl MovePicker {
    pub fn new(hash_move: Option<Move>, killers: [Move; 2]) -> Self {
        Self {
            stage: Stage::HashMove,
            hash_move,
            killers,
            next_killer: 0,
            moves: MoveList::new(),
            scores: [0; MAX_MOVES],
            next: 0,
            bad_noisy_end: 0,
            next_bad_noisy: 0,
            quiets_skipped: false,
            noisy_only: false,
        }
    }

    pub fn good_noisy_only(hash_move: Option<Move>) -> Self {
        Self {
            noisy_only: true,
            ..Self::new(hash_move, [Move::NULL; 2])
        }
    }

    pub fn skip_quiets(&mut self) {
        self.quiets_skipped = true;
    }

    pub fn next_move(
        &mut self,
        board: &Board,
        history: &History,
        continuations: &Continuations,
    ) -> Option<Move> {
        loop {
            match self.stage {
                Stage::HashMove => {
                    self.stage = Stage::GenerateNoisy;
                    if let Some(hash_move) = self.hash_move
                        && self.accepts_hash_move(board, hash_move)
                    {
                        return Some(hash_move);
                    }
                }
                Stage::GenerateNoisy => {
                    board.generate_noisy(&mut self.moves);
                    self.score_from(0, |candidate| noisy_score(board, candidate, history));
                    self.stage = Stage::GoodNoisy;
                }
                Stage::GoodNoisy => match self.select_best() {
                    Some(candidate) if Some(candidate) == self.hash_move => {}
                    Some(candidate) if board.see(candidate, 0) => return Some(candidate),
                    Some(candidate) => self.defer_bad_noisy(candidate),
                    None if self.noisy_only => self.stage = Stage::Done,
                    None => self.stage = Stage::Killers,
                },
                Stage::Killers => {
                    if let Some(killer) = self.next_valid_killer(board) {
                        return Some(killer);
                    }
                    self.stage = Stage::GenerateQuiets;
                }
                Stage::GenerateQuiets => {
                    if !self.quiets_skipped {
                        let first_quiet = self.moves.len();
                        board.generate_quiets(&mut self.moves);
                        self.score_from(first_quiet, |candidate| {
                            history.quiet_score(
                                candidate,
                                moved_piece(board, candidate),
                                continuations,
                            )
                        });
                    }
                    self.stage = Stage::Quiets;
                }
                Stage::Quiets => {
                    let candidate = if self.quiets_skipped {
                        None
                    } else {
                        self.select_best()
                    };
                    match candidate {
                        Some(candidate) if self.already_tried_quiet(candidate) => {}
                        Some(candidate) => return Some(candidate),
                        None => self.stage = Stage::BadNoisy,
                    }
                }
                Stage::BadNoisy => {
                    if self.next_bad_noisy == self.bad_noisy_end {
                        self.stage = Stage::Done;
                        continue;
                    }
                    let candidate = self.moves.as_slice()[self.next_bad_noisy];
                    self.next_bad_noisy += 1;
                    return Some(candidate);
                }
                Stage::Done => return None,
            }
        }
    }

    fn accepts_hash_move(&self, board: &Board, hash_move: Move) -> bool {
        if !board.is_pseudo_legal(hash_move) {
            return false;
        }
        !self.noisy_only || (is_noisy(hash_move) && board.see(hash_move, 0))
    }

    fn next_valid_killer(&mut self, board: &Board) -> Option<Move> {
        while self.next_killer < self.killers.len() {
            let killer = self.killers[self.next_killer];
            self.next_killer += 1;
            let playable = !self.quiets_skipped
                && Some(killer) != self.hash_move
                && !self.killers[..self.next_killer - 1].contains(&killer)
                && !is_noisy(killer)
                && board.is_pseudo_legal(killer);
            if playable {
                return Some(killer);
            }
        }
        None
    }

    fn already_tried_quiet(&self, candidate: Move) -> bool {
        Some(candidate) == self.hash_move || self.killers.contains(&candidate)
    }

    fn defer_bad_noisy(&mut self, candidate: Move) {
        if !self.noisy_only {
            self.moves.as_mut_slice()[self.bad_noisy_end] = candidate;
            self.bad_noisy_end += 1;
        }
    }

    fn score_from(&mut self, first: usize, score: impl Fn(Move) -> i32) {
        for index in first..self.moves.len() {
            self.scores[index] = score(self.moves.as_slice()[index]);
        }
    }

    fn select_best(&mut self) -> Option<Move> {
        let moves = self.moves.as_mut_slice();
        let best = (self.next..moves.len()).max_by_key(|&index| self.scores[index])?;
        moves.swap(self.next, best);
        self.scores.swap(self.next, best);
        self.next += 1;
        Some(moves[self.next - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::testing::{board, find_move};
    use crate::board::{KIWIPETE, START_POSITION};

    fn picked(board: &Board, mut picker: MovePicker, skip_after: Option<usize>) -> Vec<Move> {
        let history = History::default();
        let mut moves = Vec::new();
        if skip_after == Some(0) {
            picker.skip_quiets();
        }
        while let Some(candidate) = picker.next_move(board, &history, &[None; 2]) {
            moves.push(candidate);
            if Some(moves.len()) == skip_after {
                picker.skip_quiets();
            }
        }
        moves
    }

    fn sorted(mut moves: Vec<Move>) -> Vec<Move> {
        moves.sort_by_key(|candidate| format!("{candidate:?}"));
        moves
    }

    fn generated(board: &Board) -> Vec<Move> {
        let mut moves = MoveList::new();
        board.generate_pseudo_legal(&mut moves);
        moves.as_slice().to_vec()
    }

    #[test]
    fn yields_every_pseudo_legal_move_once_in_stage_order() {
        let cases = [
            (START_POSITION, None, [None, None]),
            (START_POSITION, Some("g1f3"), [Some("e2e4"), Some("e2e4")]),
            (KIWIPETE, Some("e2a6"), [Some("a2a3"), Some("e1g1")]),
            (KIWIPETE, Some("d5e6"), [Some("g2h3"), Some("b1b3")]),
            (KIWIPETE, Some("a1a8"), [Some("h1h8"), Some("e5c6")]),
            (
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                Some("b4c5"),
                [Some("c4c5"), Some("d1e2")],
            ),
        ];

        for (fen, hash_notation, killer_notations) in cases {
            let board = board(fen);
            let all = generated(&board);
            let lookup = |notation: &str| {
                all.iter()
                    .copied()
                    .find(|candidate| candidate.to_string() == notation)
                    .unwrap_or(Move::new(
                        notation[..2].parse().expect("square"),
                        notation[2..4].parse().expect("square"),
                        MoveKind::Quiet,
                    ))
            };
            let hash_move = hash_notation.map(lookup);
            let killers = killer_notations.map(|notation| notation.map_or(Move::NULL, lookup));
            let order = picked(&board, MovePicker::new(hash_move, killers), None);

            assert_eq!(sorted(order.clone()), sorted(all.clone()), "fen {fen:?}");
            if let Some(hash_move) = hash_move.filter(|&hash_move| all.contains(&hash_move)) {
                assert_eq!(order.first(), Some(&hash_move), "fen {fen:?}: hash first");
            }
            for killer in killers.into_iter().filter(|killer| all.contains(killer)) {
                let killer_position = order
                    .iter()
                    .position(|&played| played == killer)
                    .expect("killer yielded");
                let ordinary_quiet_before_killer = order[..killer_position].iter().any(|&played| {
                    !is_noisy(played) && Some(played) != hash_move && !killers.contains(&played)
                });
                assert!(
                    !ordinary_quiet_before_killer,
                    "fen {fen:?}: killer {killer} late"
                );
            }
            let first_bad_noisy = order
                .iter()
                .position(|&candidate| is_noisy(candidate) && !board.see(candidate, 0))
                .filter(|&index| Some(order[index]) != hash_move);
            if let Some(first_bad) = first_bad_noisy {
                assert!(
                    order[first_bad..]
                        .iter()
                        .all(|&candidate| is_noisy(candidate)),
                    "fen {fen:?}: quiet after bad noisy"
                );
            }
        }
    }

    #[test]
    fn skipping_quiets_keeps_noisy_moves() {
        let cases = [(START_POSITION, 0), (KIWIPETE, 0), (KIWIPETE, 3)];

        for (fen, skip_after) in cases {
            let board = board(fen);
            let all = generated(&board);
            let order = picked(
                &board,
                MovePicker::new(None, [Move::NULL; 2]),
                Some(skip_after),
            );
            let expected_noisy: Vec<Move> = all
                .iter()
                .copied()
                .filter(|&candidate| is_noisy(candidate))
                .collect();
            let yielded_noisy: Vec<Move> = order
                .iter()
                .copied()
                .filter(|&candidate| is_noisy(candidate))
                .collect();
            assert_eq!(sorted(yielded_noisy), sorted(expected_noisy), "fen {fen:?}");
            let quiets_after_skip = order
                .iter()
                .skip(skip_after)
                .filter(|&&candidate| !is_noisy(candidate))
                .count();
            assert_eq!(quiets_after_skip, 0, "fen {fen:?}, skip after {skip_after}");
        }
    }

    #[test]
    fn good_noisy_only_yields_winning_captures() {
        let cases = [
            (KIWIPETE, None),
            (KIWIPETE, Some("e2a6")),
            (KIWIPETE, Some("a2a3")),
            ("4k3/8/8/3p4/4P3/8/8/Q3K3 w - - 0 1", Some("a1a8")),
            ("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1", Some("d1d5")),
        ];

        for (fen, hash_notation) in cases {
            let board = board(fen);
            let hash_move = hash_notation.map(|notation| find_move(&board, notation));
            let order = picked(&board, MovePicker::good_noisy_only(hash_move), None);
            let expected: Vec<Move> = generated(&board)
                .into_iter()
                .filter(|&candidate| is_noisy(candidate) && board.see(candidate, 0))
                .collect();
            assert_eq!(sorted(order.clone()), sorted(expected), "fen {fen:?}");
            if let Some(hash_move) = hash_move.filter(|&hash_move| order.contains(&hash_move)) {
                assert_eq!(order[0], hash_move, "fen {fen:?}: hash first");
            }
        }
    }
}
