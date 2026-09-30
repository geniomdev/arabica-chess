use crate::attacks::pawn_attacks;
use crate::board::Board;
use crate::board::zobrist::{castling_key, en_passant_key, side_key};
use crate::types::{Color, Move, MoveKind, Piece, Square};

pub(super) fn en_passant_victim(target: Square) -> Square {
    Square(target.0 ^ 8)
}

fn castling_rook_path(played: Move) -> Option<(Square, Square)> {
    let king = played.from().0;
    match played.kind() {
        MoveKind::KingCastle => Some((Square(king + 3), Square(king + 1))),
        MoveKind::QueenCastle => Some((Square(king - 4), Square(king - 1))),
        _ => None,
    }
}

impl Board {
    pub fn make_move(&mut self, played: Move) -> bool {
        let us = self.state.active_color;
        let them = us.opponent();
        let (from, to) = (played.from(), played.to());
        let moving = self.mailbox[from].expect("piece on origin square");
        self.history.push(self.state);

        let captured = match played.kind() {
            MoveKind::EnPassant => {
                self.remove_piece(them, Piece::Pawn, en_passant_victim(to));
                Some(Piece::Pawn)
            }
            _ if played.is_capture() => {
                let victim = self.mailbox[to].expect("piece on captured square");
                self.remove_piece(them, victim, to);
                Some(victim)
            }
            _ => None,
        };
        match played.promotion() {
            Some(promoted) => {
                self.remove_piece(us, Piece::Pawn, from);
                self.put_piece(us, promoted, to);
            }
            None => self.move_piece(us, moving, from, to),
        }
        if let Some((rook_from, rook_to)) = castling_rook_path(played) {
            self.move_piece(us, Piece::Rook, rook_from, rook_to);
        }

        let passed = Square((from.0 + to.0) / 2);
        let capturable = !(pawn_attacks(us, passed) & self.pieces_of(them, Piece::Pawn)).is_empty();
        let state = &mut self.state;
        state.zobrist_key ^= castling_key(state.castling) ^ en_passant_key(state.en_passant);
        state.castling = state.castling.after_move(from, to);
        state.en_passant = (played.kind() == MoveKind::DoublePush && capturable).then_some(passed);
        state.zobrist_key ^= castling_key(state.castling)
            ^ en_passant_key(state.en_passant)
            ^ side_key(us)
            ^ side_key(them);
        state.halfmove_clock = if moving == Piece::Pawn || captured.is_some() {
            0
        } else {
            state.halfmove_clock.saturating_add(1)
        };
        if us == Color::Black {
            state.fullmove_number += 1;
        }
        state.active_color = them;
        state.captured = captured;
        state.played_move = played;

        let legal = !self.is_king_attacked(us);
        if !legal {
            self.unmake_move();
        }
        legal
    }

    pub fn make_null_move(&mut self) {
        let us = self.state.active_color;
        let them = us.opponent();
        self.history.push(self.state);
        let state = &mut self.state;
        state.zobrist_key ^= en_passant_key(state.en_passant) ^ side_key(us) ^ side_key(them);
        state.en_passant = None;
        state.halfmove_clock = 0;
        state.active_color = them;
        state.captured = None;
        state.played_move = Move::NULL;
    }

    pub fn unmake_null_move(&mut self) {
        self.state = self.history.pop().expect("null move to unmake");
    }

    pub fn unmake_move(&mut self) {
        let undone = self.state;
        let restored = self.history.pop().expect("move to unmake");
        let us = restored.active_color;
        let them = us.opponent();
        let played = undone.played_move;
        let (from, to) = (played.from(), played.to());

        if let Some((rook_from, rook_to)) = castling_rook_path(played) {
            self.move_piece(us, Piece::Rook, rook_to, rook_from);
        }
        match played.promotion() {
            Some(promoted) => {
                self.remove_piece(us, promoted, to);
                self.put_piece(us, Piece::Pawn, from);
            }
            None => {
                let moving = self.mailbox[to].expect("piece on target square");
                self.move_piece(us, moving, to, from);
            }
        }
        match (played.kind(), undone.captured) {
            (MoveKind::EnPassant, _) => self.put_piece(them, Piece::Pawn, en_passant_victim(to)),
            (_, Some(victim)) => self.put_piece(them, victim, to),
            _ => {}
        }
        self.state = restored;
    }
}

#[cfg(test)]
mod tests {
    use crate::board::testing::{board, find_move};
    use crate::board::{KIWIPETE, MoveList, START_POSITION};

    #[test]
    fn unmake_restores_position() {
        let fens = [
            START_POSITION,
            KIWIPETE,
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "3r3k/4P3/8/8/8/8/8/K7 w - - 0 1",
            "k7/8/8/8/8/8/3p4/4R2K b - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        ];

        for fen in fens {
            let original = board(fen);
            let mut moves = MoveList::new();
            original.generate_pseudo_legal(&mut moves);
            for &candidate in moves.as_slice() {
                let mut played = original.clone();
                if played.make_move(candidate) {
                    played.unmake_move();
                }
                assert_eq!(played, original, "fen {fen:?}, move {candidate}");
                assert_eq!(played.to_fen(), fen, "fen {fen:?}, move {candidate}");
            }
        }
    }

    #[test]
    fn null_move_passes_the_turn() {
        let cases = [
            (
                START_POSITION,
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            ),
            (
                "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 5 3",
                "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3",
            ),
        ];

        for (fen, expected) in cases {
            let original = board(fen);
            let mut position = original.clone();
            position.make_null_move();
            assert_eq!(position.to_fen(), expected, "fen {fen:?}");
            assert_eq!(
                position.state.zobrist_key,
                position.compute_zobrist_key(),
                "fen {fen:?}"
            );
            position.unmake_null_move();
            assert_eq!(position, original, "fen {fen:?}");
        }
    }

    #[test]
    fn updates_state() {
        let cases = [
            (
                START_POSITION,
                "e2e4",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            ),
            (
                "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
                "e5f6",
                "rnbqkbnr/ppp1p1pp/5P2/3p4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3",
            ),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 5 10",
                "e8c8",
                "2kr3r/8/8/8/8/8/8/R3K2R w KQ - 6 11",
            ),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "a1a8",
                "R3k2r/8/8/8/8/8/8/4K2R b Kk - 0 1",
            ),
            (
                "3r3k/4P3/8/8/8/8/8/K7 w - - 0 1",
                "e7d8n",
                "3N3k/8/8/8/8/8/8/K7 b - - 0 1",
            ),
        ];

        for (fen, notation, expected) in cases {
            let mut position = board(fen);
            let candidate = find_move(&position, notation);
            assert!(
                position.make_move(candidate),
                "fen {fen:?}, move {notation}"
            );
            assert_eq!(position.to_fen(), expected, "fen {fen:?}, move {notation}");
        }
    }
}
