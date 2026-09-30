use std::sync::LazyLock;

use crate::types::{Bitboard, Color, SQUARES, Square};

type Direction = (i8, i8);

const ROOK_DIRECTIONS: [Direction; 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRECTIONS: [Direction; 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const KNIGHT_STEPS: [Direction; 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const KING_STEPS: [Direction; 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];
const SLIDER_TABLE_SIZE: usize = 102_400 + 5_248;
const MAGIC_SEEDS_BY_RANK: [u64; 8] = [728, 10316, 55013, 32803, 12281, 15100, 16645, 255];

static KNIGHT_ATTACKS: [u64; SQUARES] = leaper_table(KNIGHT_STEPS);
static KING_ATTACKS: [u64; SQUARES] = leaper_table(KING_STEPS);
static PAWN_ATTACKS: [[u64; SQUARES]; 2] = [
    leaper_table([(-1, 1), (1, 1)]),
    leaper_table([(-1, -1), (1, -1)]),
];
static SLIDERS: LazyLock<Sliders> = LazyLock::new(Sliders::build);

pub fn knight_attacks(square: Square) -> Bitboard {
    Bitboard(KNIGHT_ATTACKS[square.index()])
}

pub fn king_attacks(square: Square) -> Bitboard {
    Bitboard(KING_ATTACKS[square.index()])
}

pub fn pawn_attacks(side: Color, square: Square) -> Bitboard {
    Bitboard(PAWN_ATTACKS[side.index()][square.index()])
}

pub fn rook_attacks(square: Square, occupancy: Bitboard) -> Bitboard {
    let sliders = &*SLIDERS;
    Bitboard(sliders.attacks[sliders.rook[square.index()].index(occupancy.0)])
}

pub fn bishop_attacks(square: Square, occupancy: Bitboard) -> Bitboard {
    let sliders = &*SLIDERS;
    Bitboard(sliders.attacks[sliders.bishop[square.index()].index(occupancy.0)])
}

pub fn queen_attacks(square: Square, occupancy: Bitboard) -> Bitboard {
    rook_attacks(square, occupancy) | bishop_attacks(square, occupancy)
}

pub fn init() {
    LazyLock::force(&SLIDERS);
}

#[derive(Clone, Copy)]
struct Magic {
    mask: u64,
    factor: u64,
    shift: u32,
    offset: u32,
}

impl Magic {
    #[inline(always)]
    fn index(self, occupancy: u64) -> usize {
        let hash = (occupancy & self.mask).wrapping_mul(self.factor) >> self.shift;
        hash as usize + self.offset as usize
    }
}

struct Sliders {
    rook: [Magic; SQUARES],
    bishop: [Magic; SQUARES],
    attacks: Box<[u64]>,
}

impl Sliders {
    fn build() -> Self {
        let mut attacks = Vec::with_capacity(SLIDER_TABLE_SIZE);
        let rook = std::array::from_fn(|square| {
            find_magic(Square(square as u8), ROOK_DIRECTIONS, &mut attacks)
        });
        let bishop = std::array::from_fn(|square| {
            find_magic(Square(square as u8), BISHOP_DIRECTIONS, &mut attacks)
        });
        debug_assert_eq!(attacks.len(), SLIDER_TABLE_SIZE);
        Self {
            rook,
            bishop,
            attacks: attacks.into_boxed_slice(),
        }
    }
}

fn find_magic(square: Square, directions: [Direction; 4], attacks: &mut Vec<u64>) -> Magic {
    let mut random = XorShift(MAGIC_SEEDS_BY_RANK[square.rank() as usize]);
    let mask = relevant_mask(square, directions);
    let occupancies: Vec<u64> = subsets(mask).collect();
    let references: Vec<u64> = occupancies
        .iter()
        .map(|&occupancy| sliding_attacks(square, occupancy, directions))
        .collect();
    let offset = attacks.len();
    attacks.resize(offset + occupancies.len(), 0);
    let table = &mut attacks[offset..];
    let mut used_in_attempt = vec![0u32; table.len()];
    let mut attempt = 0;
    loop {
        attempt += 1;
        let factor = random.sparse();
        if (mask.wrapping_mul(factor) >> 56).count_ones() < 6 {
            continue;
        }
        let magic = Magic {
            mask,
            factor,
            shift: 64 - mask.count_ones(),
            offset: 0,
        };
        let collision_free = occupancies
            .iter()
            .zip(&references)
            .all(|(&occupancy, &reference)| {
                let slot = magic.index(occupancy);
                if used_in_attempt[slot] != attempt {
                    used_in_attempt[slot] = attempt;
                    table[slot] = reference;
                    true
                } else {
                    table[slot] == reference
                }
            });
        if collision_free {
            return Magic {
                offset: offset as u32,
                ..magic
            };
        }
    }
}

fn relevant_mask(square: Square, directions: [Direction; 4]) -> u64 {
    let rank_edges = (Bitboard::rank(0) | Bitboard::rank(7)) & !Bitboard::rank(square.rank());
    let file_edges = (Bitboard::file(0) | Bitboard::file(7)) & !Bitboard::file(square.file());
    sliding_attacks(square, 0, directions) & !(rank_edges | file_edges).0
}

fn sliding_attacks(square: Square, occupancy: u64, directions: [Direction; 4]) -> u64 {
    let mut attacks = 0;
    for (file_step, rank_step) in directions {
        let mut file = square.file() as i8 + file_step;
        let mut rank = square.rank() as i8 + rank_step;
        while (0..8).contains(&file) && (0..8).contains(&rank) {
            let bit = 1u64 << (rank * 8 + file);
            attacks |= bit;
            if occupancy & bit != 0 {
                break;
            }
            file += file_step;
            rank += rank_step;
        }
    }
    attacks
}

fn subsets(mask: u64) -> impl Iterator<Item = u64> {
    let mut next = Some(0u64);
    std::iter::from_fn(move || {
        let current = next?;
        let following = current.wrapping_sub(mask) & mask;
        next = (following != 0).then_some(following);
        Some(current)
    })
}

const fn leaper_table<const N: usize>(steps: [Direction; N]) -> [u64; SQUARES] {
    let mut table = [0u64; SQUARES];
    let mut square = 0;
    while square < SQUARES {
        let mut step = 0;
        while step < N {
            let file = (square % 8) as i8 + steps[step].0;
            let rank = (square / 8) as i8 + steps[step].1;
            if file >= 0 && file < 8 && rank >= 0 && rank < 8 {
                table[square] |= 1u64 << (rank * 8 + file);
            }
            step += 1;
        }
        square += 1;
    }
    table
}

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::testing::square;

    #[test]
    fn magic_lookup_matches_ray_walk() {
        let mut random = XorShift(42);
        for index in 0..SQUARES as u8 {
            let square = Square(index);
            for _ in 0..1000 {
                let occupancy = random.next() & random.next();
                let board = Bitboard(occupancy);
                assert_eq!(
                    rook_attacks(square, board).0,
                    sliding_attacks(square, occupancy, ROOK_DIRECTIONS)
                );
                assert_eq!(
                    bishop_attacks(square, board).0,
                    sliding_attacks(square, occupancy, BISHOP_DIRECTIONS)
                );
            }
        }
    }

    #[test]
    fn leaper_tables_match_targets() {
        let cases: [(&str, Bitboard, &[&str]); 6] = [
            ("knight a1", knight_attacks(square("a1")), &["b3", "c2"]),
            (
                "knight d4",
                knight_attacks(square("d4")),
                &["b3", "b5", "c2", "c6", "e2", "e6", "f3", "f5"],
            ),
            ("king a1", king_attacks(square("a1")), &["a2", "b1", "b2"]),
            (
                "king d4",
                king_attacks(square("d4")),
                &["c3", "c4", "c5", "d3", "d5", "e3", "e4", "e5"],
            ),
            (
                "white pawn a2",
                pawn_attacks(Color::White, square("a2")),
                &["b3"],
            ),
            (
                "black pawn h7",
                pawn_attacks(Color::Black, square("h7")),
                &["g6"],
            ),
        ];

        for (name, actual, targets) in cases {
            let mut expected = Bitboard::empty();
            for &target in targets {
                expected |= Bitboard::from_square(square(target));
            }
            assert_eq!(actual, expected, "{name}");
        }
    }
}
