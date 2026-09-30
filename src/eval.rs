use crate::board::Board;
use crate::types::{Color, EveryPiece, Piece, SQUARES, Square};

const PIECE_VALUES: [i32; 6] = [0, 900, 500, 330, 320, 100];

#[rustfmt::skip]
const PIECE_SQUARE_TABLES: [[i32; SQUARES]; 6] = [
    [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
         20,  20,   0,   0,   0,   0,  20,  20,
         20,  30,  10,   0,   0,  10,  30,  20,
    ],
    [
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,   5,   5,   5,   0, -10,
         -5,   0,   5,   5,   5,   5,   0,  -5,
          0,   0,   5,   5,   5,   5,   0,  -5,
        -10,   5,   5,   5,   5,   5,   0, -10,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20,
    ],
    [
          0,   0,   0,   0,   0,   0,   0,   0,
          5,  10,  10,  10,  10,  10,  10,   5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
          0,   0,   0,   5,   5,   0,   0,   0,
    ],
    [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -30,   0,  10,  15,  15,  10,   0, -30,
        -30,   5,  15,  20,  20,  15,   5, -30,
        -30,   0,  15,  20,  20,  15,   0, -30,
        -30,   5,  10,  15,  15,  10,   5, -30,
        -40, -20,   0,   5,   5,   0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    [
          0,   0,   0,   0,   0,   0,   0,   0,
         50,  50,  50,  50,  50,  50,  50,  50,
         10,  10,  20,  30,  30,  20,  10,  10,
          5,   5,  10,  25,  25,  10,   5,   5,
          0,   0,   0,  20,  20,   0,   0,   0,
          5,  -5, -10,   0,   0, -10,  -5,   5,
          5,  10,  10, -20, -20,  10,  10,   5,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
];

static PIECE_SQUARE_SCORES: EveryPiece<[[i32; SQUARES]; 2]> = build_piece_square_scores();

const fn build_piece_square_scores() -> EveryPiece<[[i32; SQUARES]; 2]> {
    let mut scores = [[[0; SQUARES]; 2]; 6];
    let mut piece = 0;
    while piece < 6 {
        let mut square = 0;
        while square < SQUARES {
            let value = PIECE_VALUES[piece];
            scores[piece][Color::White as usize][square] =
                value + PIECE_SQUARE_TABLES[piece][square ^ 56];
            scores[piece][Color::Black as usize][square] =
                value + PIECE_SQUARE_TABLES[piece][square];
            square += 1;
        }
        piece += 1;
    }
    EveryPiece::from_array(scores)
}

pub const fn piece_value(piece: Piece) -> i32 {
    PIECE_VALUES[piece.index()]
}

fn side_score(board: &Board, side: Color) -> i32 {
    Piece::ALL
        .into_iter()
        .map(|piece| {
            let table = &PIECE_SQUARE_SCORES[piece][side.index()];
            board
                .pieces_of(side, piece)
                .into_iter()
                .map(|square: Square| table[square.index()])
                .sum::<i32>()
        })
        .sum()
}

pub fn evaluate(board: &Board) -> i32 {
    let us = board.state.active_color;
    side_score(board, us) - side_score(board, us.opponent())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{KIWIPETE, START_POSITION};

    #[test]
    fn evaluates_positions() {
        let cases = [
            (START_POSITION, 0..=0),
            ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 850..=950),
            ("4k3/8/8/8/8/8/8/3QK3 b - - 0 1", -950..=-850),
            ("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1", 0..=0),
        ];

        for (fen, range) in cases {
            let board: Board = fen
                .parse()
                .unwrap_or_else(|error| panic!("fen {fen:?} rejected: {error}"));
            let score = evaluate(&board);
            assert!(range.contains(&score), "fen {fen:?}, score {score}");
        }
    }

    #[test]
    fn mirrored_positions_evaluate_equally() {
        let cases = [
            (
                KIWIPETE,
                "r3k2r/pppbbppp/2n2q1P/1P2p3/3pn3/BN2PNP1/P1PPQPB1/R3K2R b KQkq - 0 1",
            ),
            (
                "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1",
                "4k3/8/8/4p3/8/8/8/4K3 b - - 0 1",
            ),
            (
                "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 3",
            ),
        ];

        for (fen, mirrored) in cases {
            let [score, mirrored_score] = [fen, mirrored].map(|fen| {
                let board: Board = fen
                    .parse()
                    .unwrap_or_else(|error| panic!("fen {fen:?} rejected: {error}"));
                evaluate(&board)
            });
            assert_eq!(score, mirrored_score, "fen {fen:?} vs {mirrored:?}");
        }
    }
}
