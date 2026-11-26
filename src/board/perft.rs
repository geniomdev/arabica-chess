use crate::board::{Board, MoveList};

impl Board {
    pub fn perft(&mut self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let mut moves = MoveList::new();
        self.generate_pseudo_legal(&mut moves);
        let mut nodes = 0;
        for &candidate in moves.as_slice() {
            if self.make_move(candidate) {
                nodes += self.perft(depth - 1);
                self.unmake_move();
            }
        }
        nodes
    }
}

#[cfg(test)]
mod tests {
    use crate::board::testing::board;
    use crate::board::{KIWIPETE, START_POSITION};

    #[test]
    fn matches_reference_counts() {
        let cases: [(&str, &[u64]); 10] = [
            (START_POSITION, &[20, 400, 8_902, 197_281]),
            (KIWIPETE, &[48, 2_039, 97_862]),
            (
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                &[14, 191, 2_812, 43_238],
            ),
            (
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                &[6, 264, 9_467],
            ),
            (
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                &[44, 1_486, 62_379],
            ),
            (
                "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
                &[21, 528, 12_189],
            ),
            (
                "2nnrbkr/p1qppppp/8/1ppb4/6PP/3PP3/PPP2P2/BQNNRBKR w HEhe - 1 9",
                &[21, 807, 18_002],
            ),
            ("4k3/8/8/8/8/8/8/qRK4R w BH - 0 1", &[16, 265, 5_087]),
            ("1r4kr/8/8/8/8/8/8/1R4KR w BHbh - 0 1", &[24, 479, 10_954]),
            ("rk5r/8/8/8/8/8/8/RK5R w AHah - 0 1", &[24, 479, 11_099]),
        ];

        for (fen, expected) in cases {
            let mut board = board(fen);
            for (depth, &nodes) in (1..).zip(expected) {
                assert_eq!(board.perft(depth), nodes, "fen {fen:?}, depth {depth}");
            }
        }
    }
}
