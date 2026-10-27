use crate::attacks::{bishop_attacks, rook_attacks};
use crate::board::Board;
use crate::board::makemove::en_passant_victim;
use crate::types::{Bitboard, Color, Move, MoveKind, Piece, Square};

const SEE_VALUES: [i32; 6] = [20_000, 900, 500, 330, 320, 100];
const MAX_PINS: usize = 16;

type SliderAttacks = fn(Square, Bitboard) -> Bitboard;

#[derive(Clone, Copy, Default)]
struct Pin {
    pinned: Bitboard,
    pinner: Bitboard,
}

#[derive(Default)]
struct Pins {
    list: [Pin; MAX_PINS],
    len: usize,
}

impl Pins {
    fn frozen(&self, occupancy: Bitboard) -> Bitboard {
        self.list[..self.len]
            .iter()
            .filter(|pin| !(pin.pinner & occupancy).is_empty())
            .fold(Bitboard::empty(), |frozen, pin| frozen | pin.pinned)
    }
}

pub const fn see_value(piece: Piece) -> i32 {
    SEE_VALUES[piece.index()]
}

impl Board {
    pub fn see(&self, played: Move, threshold: i32) -> bool {
        if matches!(played.kind(), MoveKind::KingCastle | MoveKind::QueenCastle) {
            return threshold <= 0;
        }
        let (from, to) = (played.from(), played.to());
        let captured_value = match played.kind() {
            MoveKind::EnPassant => see_value(Piece::Pawn),
            _ => self.piece_on(to).map_or(0, see_value),
        };
        let mut swap = captured_value - threshold;
        if swap < 0 {
            return false;
        }
        let moving = self.piece_on(from).expect("piece on origin square");
        swap = see_value(moving) - swap;
        if swap <= 0 {
            return true;
        }

        let mut occupancy =
            self.occupancy() & !Bitboard::from_square(from) & !Bitboard::from_square(to);
        if played.kind() == MoveKind::EnPassant {
            occupancy &= !Bitboard::from_square(en_passant_victim(to));
        }
        let diagonal = self.bishops_and_queens();
        let orthogonal = self.rooks_and_queens();
        let pins = self.pins_off_line(from, to);
        let mut attackers = self.attackers_to(to, occupancy);
        let mut side = self.state.active_color;
        let mut winning = true;

        loop {
            side = side.opponent();
            attackers &= occupancy;
            let side_attackers = attackers & self.occupied_by(side) & !pins.frozen(occupancy);
            if side_attackers.is_empty() {
                break;
            }
            winning = !winning;
            let (piece, candidates) = Piece::CHEAPEST_FIRST
                .into_iter()
                .map(|piece| (piece, side_attackers & self.pieces(piece)))
                .find(|(_, candidates)| !candidates.is_empty())
                .expect("attacker of some piece type");
            if piece == Piece::King {
                let defended = !(attackers & self.occupied_by(side.opponent())).is_empty();
                return if defended { !winning } else { winning };
            }
            swap = see_value(piece) - swap;
            if swap < i32::from(winning) {
                break;
            }
            occupancy &= !Bitboard::from_square(candidates.lowest_square());
            if matches!(piece, Piece::Pawn | Piece::Bishop | Piece::Queen) {
                attackers |= bishop_attacks(to, occupancy) & diagonal;
            }
            if matches!(piece, Piece::Rook | Piece::Queen) {
                attackers |= rook_attacks(to, occupancy) & orthogonal;
            }
        }
        winning
    }

    fn pins_off_line(&self, from: Square, to: Square) -> Pins {
        let moved = Bitboard::from_square(from);
        let landed = Bitboard::from_square(to);
        let occupancy = (self.occupancy() & !moved) | landed;
        let mut pins = Pins::default();
        for side in Color::ALL {
            let king = self.king_square(side);
            let enemy = side.opponent();
            let rays: [(SliderAttacks, Bitboard); 2] = [
                (rook_attacks, self.orthogonal_sliders(enemy)),
                (bishop_attacks, self.diagonal_sliders(enemy)),
            ];
            for (slider_attacks, sliders) in rays {
                let enemy_pieces = self.occupied_by(enemy) & !moved;
                for pinner in slider_attacks(king, enemy_pieces) & sliders & !moved {
                    let pinned = slider_attacks(king, occupancy)
                        & slider_attacks(pinner, occupancy)
                        & occupancy
                        & self.occupied_by(side)
                        & !landed;
                    if pinned.is_empty() {
                        continue;
                    }
                    let unpinned = occupancy & !pinned;
                    let pin_line =
                        slider_attacks(king, unpinned) & slider_attacks(pinner, unpinned);
                    if pinner == to || pin_line.contains(to) {
                        continue;
                    }
                    pins.list[pins.len] = Pin {
                        pinned,
                        pinner: Bitboard::from_square(pinner),
                    };
                    pins.len += 1;
                }
            }
        }
        pins
    }
}

#[cfg(test)]
mod tests {
    use crate::board::testing::{board, find_move};

    #[test]
    fn static_exchange_evaluation() {
        let cases = [
            ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 0, true),
            ("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 0, true),
            ("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1", "d1d5", 0, false),
            ("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1", "d1d5", -800, true),
            ("4k3/8/8/3r4/8/8/3R4/3RK3 w - - 0 1", "d2d5", 0, true),
            ("3rk3/8/8/3r4/8/8/3R4/4K3 w - - 0 1", "d2d5", 0, true),
            ("3rk3/8/8/3r4/8/8/3R4/4K3 w - - 0 1", "d2d5", 1, false),
            ("4k3/8/8/3n4/8/4B3/8/4K3 w - - 0 1", "e3d4", 0, true),
            ("4k3/8/4p3/3n4/8/8/8/3QK3 w - - 0 1", "d1d5", 0, false),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 0, true),
            ("4k3/8/2b5/8/8/8/6Q1/4K2B w - - 0 1", "g2c6", 0, true),
            ("4k3/1b6/2b5/8/8/8/6Q1/4K3 w - - 0 1", "g2c6", 0, false),
            ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1g1", 0, true),
            ("4k3/4r3/8/1N6/8/8/8/4R1K1 w - - 0 1", "b5c7", 0, true),
            ("4k3/4r3/8/8/7Q/8/8/4R1K1 w - - 0 1", "h4e4", 0, false),
            ("4k3/4r3/8/1N6/8/8/8/4R1K1 b - - 0 1", "e7e1", 0, true),
            ("4k3/4r3/2b5/8/8/8/8/4R1K1 w - - 0 1", "e1e7", 0, true),
            (
                "6k1/pp1r1pp1/2p1p2p/4P2P/3q1Q2/1P4R1/P1Pr1PP1/R5K1 w - - 0 24",
                "f4f6",
                0,
                true,
            ),
            (
                "6k1/pp1r1pp1/2p1p2p/4P2P/3q1Q2/1P6/P1Pr1PP1/R5K1 w - - 0 24",
                "f4f6",
                0,
                false,
            ),
        ];

        for (fen, notation, threshold, expected) in cases {
            let board = board(fen);
            let played = find_move(&board, notation);
            assert_eq!(
                board.see(played, threshold),
                expected,
                "fen {fen:?}, move {notation}, threshold {threshold}"
            );
        }
    }
}
