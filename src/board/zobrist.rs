use super::Board;
use crate::types::{CastlingRights, Color, FILES, PIECE_TYPES, Piece, SIDES, SQUARES, Square};

const SEED: u64 = 0x2545_F491_4F6C_DD1D;
const CASTLING_COMBINATIONS: usize = CastlingRights::ALL.index() + 1;

struct Keys {
    pieces: [[[u64; SQUARES]; PIECE_TYPES]; SIDES],
    castling: [u64; CASTLING_COMBINATIONS],
    en_passant_file: [u64; FILES as usize],
    black_to_move: u64,
}

static KEYS: Keys = generate_keys();

const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

const fn generate_keys() -> Keys {
    let mut state = SEED;
    let mut pieces = [[[0; SQUARES]; PIECE_TYPES]; SIDES];
    let mut side = 0;
    while side < SIDES {
        let mut piece = 0;
        while piece < PIECE_TYPES {
            let mut square = 0;
            while square < SQUARES {
                pieces[side][piece][square] = splitmix64(&mut state);
                square += 1;
            }
            piece += 1;
        }
        side += 1;
    }

    let mut single_right_keys = [0; 4];
    let mut right = 0;
    while right < single_right_keys.len() {
        single_right_keys[right] = splitmix64(&mut state);
        right += 1;
    }
    let mut castling = [0; CASTLING_COMBINATIONS];
    let mut rights = 0;
    while rights < CASTLING_COMBINATIONS {
        let mut right = 0;
        while right < single_right_keys.len() {
            if rights & (1 << right) != 0 {
                castling[rights] ^= single_right_keys[right];
            }
            right += 1;
        }
        rights += 1;
    }

    let mut en_passant_file = [0; FILES as usize];
    let mut file = 0;
    while file < en_passant_file.len() {
        en_passant_file[file] = splitmix64(&mut state);
        file += 1;
    }

    Keys {
        pieces,
        castling,
        en_passant_file,
        black_to_move: splitmix64(&mut state),
    }
}

pub fn piece_key(side: Color, piece: Piece, square: Square) -> u64 {
    KEYS.pieces[side.index()][piece.index()][square.index()]
}

pub fn castling_key(rights: CastlingRights) -> u64 {
    KEYS.castling[rights.index()]
}

pub fn en_passant_key(square: Option<Square>) -> u64 {
    square.map_or(0, |square| KEYS.en_passant_file[square.file() as usize])
}

pub fn side_key(side: Color) -> u64 {
    match side {
        Color::White => 0,
        Color::Black => KEYS.black_to_move,
    }
}

impl Board {
    pub fn compute_zobrist_key(&self) -> u64 {
        let mut key = castling_key(self.state.castling)
            ^ en_passant_key(self.state.en_passant)
            ^ side_key(self.state.active_color);
        for side in Color::ALL {
            for piece in Piece::ALL {
                for square in self.pieces_of(side, piece) {
                    key ^= piece_key(side, piece, square);
                }
            }
        }
        key
    }

    pub fn is_repetition(&self, search_ply: usize) -> bool {
        let key = self.state.zobrist_key;
        let states = &self.history;
        let reversible_plies = usize::from(self.state.halfmove_clock).min(states.len());
        let mut earlier_occurrences = 0;
        for distance in (4..=reversible_plies).step_by(2) {
            if states[states.len() - distance].zobrist_key != key {
                continue;
            }
            if distance <= search_ply {
                return true;
            }
            earlier_occurrences += 1;
            if earlier_occurrences == 2 {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use crate::board::testing::{board, find_move};
    use crate::board::{Board, KIWIPETE, MoveList, START_POSITION};

    fn play(board: &mut Board, notation: &str) {
        let candidate = find_move(board, notation);
        assert!(board.make_move(candidate), "move {notation} is illegal");
    }

    #[test]
    fn incremental_key_matches_full_computation() {
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
            assert_eq!(
                original.state.zobrist_key,
                original.compute_zobrist_key(),
                "fen {fen:?}"
            );
            let mut moves = MoveList::new();
            original.generate_pseudo_legal(&mut moves);
            for &candidate in moves.as_slice() {
                let mut played = original.clone();
                if !played.make_move(candidate) {
                    continue;
                }
                assert_eq!(
                    played.state.zobrist_key,
                    played.compute_zobrist_key(),
                    "fen {fen:?}, move {candidate}"
                );
                assert_eq!(
                    played.state.zobrist_key,
                    board(&played.to_fen()).state.zobrist_key,
                    "fen {fen:?}, move {candidate}"
                );
            }
        }
    }

    #[test]
    fn transpositions_share_key() {
        let cases = [
            (START_POSITION, "g1f3 g8f6 b1c3", "b1c3 g8f6 g1f3"),
            (START_POSITION, "e2e4 e7e6 d2d4", "d2d4 e7e6 e2e4"),
            (KIWIPETE, "e1g1 e8c8", "e1g1 e8c8"),
        ];

        for (fen, first, second) in cases {
            let mut first_board = board(fen);
            let mut second_board = board(fen);
            first
                .split_whitespace()
                .for_each(|notation| play(&mut first_board, notation));
            second
                .split_whitespace()
                .for_each(|notation| play(&mut second_board, notation));
            assert_eq!(
                first_board.state.zobrist_key, second_board.state.zobrist_key,
                "fen {fen:?}, {first:?} vs {second:?}"
            );
        }
    }

    #[test]
    fn detects_repetitions() {
        let shuffle = "g1f3 g8f6 f3g1 f6g8";
        let cases = [
            (START_POSITION, "", 0, false),
            (START_POSITION, "g1f3 g8f6 f3g1", 0, false),
            (START_POSITION, shuffle, 0, false),
            (START_POSITION, shuffle, 4, true),
            (START_POSITION, &format!("{shuffle} {shuffle}"), 0, true),
            (
                START_POSITION,
                &format!("{shuffle} e2e4 e7e5 {shuffle}"),
                0,
                false,
            ),
            (START_POSITION, "e2e4 e7e5 g1f3 g8f6 f3g1 f6g8", 4, true),
            (
                "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                "e1d1 e8d8 d1e1 d8e8",
                4,
                true,
            ),
        ];

        for (fen, line, search_ply, expected) in cases {
            let mut position = board(fen);
            line.split_whitespace()
                .for_each(|notation| play(&mut position, notation));
            assert_eq!(
                position.is_repetition(search_ply),
                expected,
                "fen {fen:?}, line {line:?}, search ply {search_ply}"
            );
        }
    }
}
