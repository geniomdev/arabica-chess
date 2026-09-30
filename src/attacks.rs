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

const ROOK_FACTORS: [u64; SQUARES] = [
    0x0A80004000801220,
    0x8040004010002008,
    0x2080200010008008,
    0x1100100008210004,
    0xC200209084020008,
    0x2100010004000208,
    0x0400081000822421,
    0x0200010422048844,
    0x0800800080400024,
    0x0001402000401000,
    0x3000801000802001,
    0x4400800800100083,
    0x0904802402480080,
    0x4040800400020080,
    0x0018808042000100,
    0x4040800080004100,
    0x0040048001458024,
    0x00A0004000205000,
    0x3100808010002000,
    0x4825010010000820,
    0x5004808008000401,
    0x2024818004000A00,
    0x0005808002000100,
    0x2100060004806104,
    0x0080400880008421,
    0x4062220600410280,
    0x010A004A00108022,
    0x0000100080080080,
    0x0021000500080010,
    0x0044000202001008,
    0x0000100400080102,
    0xC020128200040545,
    0x0080002000400040,
    0x0000804000802004,
    0x0000120022004080,
    0x010A386103001001,
    0x9010080080800400,
    0x8440020080800400,
    0x0004228824001001,
    0x000000490A000084,
    0x0080002000504000,
    0x200020005000C000,
    0x0012088020420010,
    0x0010010080080800,
    0x0085001008010004,
    0x0002000204008080,
    0x0040413002040008,
    0x0000304081020004,
    0x0080204000800080,
    0x3008804000290100,
    0x1010100080200080,
    0x2008100208028080,
    0x5000850800910100,
    0x8402019004680200,
    0x0120911028020400,
    0x0000008044010200,
    0x0020850200244012,
    0x0020850200244012,
    0x0000102001040841,
    0x140900040A100021,
    0x000200282410A102,
    0x000200282410A102,
    0x000200282410A102,
    0x4048240043802106,
];
const BISHOP_FACTORS: [u64; SQUARES] = [
    0x40106000A1160020,
    0x0020010250810120,
    0x2010010220280081,
    0x002806004050C040,
    0x0002021018000000,
    0x2001112010000400,
    0x0881010120218080,
    0x1030820110010500,
    0x0000120222042400,
    0x2000020404040044,
    0x8000480094208000,
    0x0003422A02000001,
    0x000A220210100040,
    0x8004820202226000,
    0x0018234854100800,
    0x0100004042101040,
    0x0004001004082820,
    0x0010000810010048,
    0x1014004208081300,
    0x2080818802044202,
    0x0040880C00A00100,
    0x0080400200522010,
    0x0001000188180B04,
    0x0080249202020204,
    0x1004400004100410,
    0x00013100A0022206,
    0x2148500001040080,
    0x4241080011004300,
    0x4020848004002000,
    0x10101380D1004100,
    0x0008004422020284,
    0x01010A1041008080,
    0x0808080400082121,
    0x0808080400082121,
    0x0091128200100C00,
    0x0202200802010104,
    0x8C0A020200440085,
    0x01A0008080B10040,
    0x0889520080122800,
    0x100902022202010A,
    0x04081A0816002000,
    0x0000681208005000,
    0x8170840041008802,
    0x0A00004200810805,
    0x0830404408210100,
    0x2602208106006102,
    0x1048300680802628,
    0x2602208106006102,
    0x0602010120110040,
    0x0941010801043000,
    0x000040440A210428,
    0x0008240020880021,
    0x0400002012048200,
    0x00AC102001210220,
    0x0220021002009900,
    0x84440C080A013080,
    0x0001008044200440,
    0x0004C04410841000,
    0x2000500104011130,
    0x1A0C010011C20229,
    0x0044800112202200,
    0x0434804908100424,
    0x0300404822C08200,
    0x48081010008A2A80,
];

static KNIGHT_ATTACKS: [u64; SQUARES] = leaper_table(KNIGHT_STEPS);
static KING_ATTACKS: [u64; SQUARES] = leaper_table(KING_STEPS);
static PAWN_ATTACKS: [[u64; SQUARES]; 2] = [
    leaper_table([(-1, 1), (1, 1)]),
    leaper_table([(-1, -1), (1, -1)]),
];
#[allow(long_running_const_eval)]
static SLIDERS: Sliders = Sliders::build();

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
    Bitboard(SLIDERS.attacks[SLIDERS.rook[square.index()].index(occupancy.0)])
}

pub fn bishop_attacks(square: Square, occupancy: Bitboard) -> Bitboard {
    Bitboard(SLIDERS.attacks[SLIDERS.bishop[square.index()].index(occupancy.0)])
}

pub fn queen_attacks(square: Square, occupancy: Bitboard) -> Bitboard {
    rook_attacks(square, occupancy) | bishop_attacks(square, occupancy)
}

#[derive(Clone, Copy)]
struct Magic {
    mask: u64,
    factor: u64,
    shift: u32,
    offset: u32,
}

impl Magic {
    const EMPTY: Self = Self {
        mask: 0,
        factor: 0,
        shift: 0,
        offset: 0,
    };

    #[inline(always)]
    const fn index(self, occupancy: u64) -> usize {
        let hash = (occupancy & self.mask).wrapping_mul(self.factor) >> self.shift;
        hash as usize + self.offset as usize
    }
}

struct Sliders {
    rook: [Magic; SQUARES],
    bishop: [Magic; SQUARES],
    attacks: [u64; SLIDER_TABLE_SIZE],
}

impl Sliders {
    const fn build() -> Self {
        let mut sliders = Self {
            rook: [Magic::EMPTY; SQUARES],
            bishop: [Magic::EMPTY; SQUARES],
            attacks: [0; SLIDER_TABLE_SIZE],
        };
        let mut offset = 0;
        let mut square = 0;
        while square < SQUARES {
            let magic = sliders.fill(square, ROOK_DIRECTIONS, ROOK_FACTORS[square], offset);
            offset += 1 << magic.mask.count_ones();
            sliders.rook[square] = magic;
            square += 1;
        }
        square = 0;
        while square < SQUARES {
            let magic = sliders.fill(square, BISHOP_DIRECTIONS, BISHOP_FACTORS[square], offset);
            offset += 1 << magic.mask.count_ones();
            sliders.bishop[square] = magic;
            square += 1;
        }
        assert!(offset == SLIDER_TABLE_SIZE);
        sliders
    }

    const fn fill(
        &mut self,
        square: usize,
        directions: [Direction; 4],
        factor: u64,
        offset: usize,
    ) -> Magic {
        let square = Square::from_index(square as u8);
        let mask = relevant_mask(square, directions);
        let magic = Magic {
            mask,
            factor,
            shift: 64 - mask.count_ones(),
            offset: offset as u32,
        };
        let mut occupancy = 0;
        loop {
            let reference = sliding_attacks(square, occupancy, directions);
            let slot = magic.index(occupancy);
            assert!(
                self.attacks[slot] == 0 || self.attacks[slot] == reference,
                "magic factor collides"
            );
            self.attacks[slot] = reference;
            occupancy = occupancy.wrapping_sub(mask) & mask;
            if occupancy == 0 {
                return magic;
            }
        }
    }
}

const fn relevant_mask(square: Square, directions: [Direction; 4]) -> u64 {
    let rank_edges = (Bitboard::rank(0).0 | Bitboard::rank(7).0) & !Bitboard::rank(square.rank()).0;
    let file_edges = (Bitboard::file(0).0 | Bitboard::file(7).0) & !Bitboard::file(square.file()).0;
    sliding_attacks(square, 0, directions) & !(rank_edges | file_edges)
}

const fn sliding_attacks(square: Square, occupancy: u64, directions: [Direction; 4]) -> u64 {
    let mut attacks = 0;
    let mut direction = 0;
    while direction < directions.len() {
        let (file_step, rank_step) = directions[direction];
        let mut file = square.file() as i8 + file_step;
        let mut rank = square.rank() as i8 + rank_step;
        while file >= 0 && file < 8 && rank >= 0 && rank < 8 {
            let bit = 1u64 << (rank * 8 + file);
            attacks |= bit;
            if occupancy & bit != 0 {
                break;
            }
            file += file_step;
            rank += rank_step;
        }
        direction += 1;
    }
    attacks
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::testing::square;

    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    #[test]
    fn magic_lookup_matches_ray_walk() {
        let mut random = XorShift(42);
        for index in 0..SQUARES as u8 {
            let square = Square::from_index(index);
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
