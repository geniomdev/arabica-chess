pub const SIDES: usize = 2;
pub const PIECE_TYPES: usize = 6;
pub const SQUARES: usize = 64;
pub const FILES: u8 = 8;
pub const RANKS: u8 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Square(pub u8);

impl Square {
    pub const fn new(file: u8, rank: u8) -> Self {
        debug_assert!(file < FILES && rank < RANKS);
        Self(rank * FILES + file)
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn file(self) -> u8 {
        self.0 % FILES
    }

    pub const fn rank(self) -> u8 {
        self.0 / FILES
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseSquareError;

impl std::fmt::Display for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        write!(f, "{file}{rank}")
    }
}

impl std::str::FromStr for Square {
    type Err = ParseSquareError;

    fn from_str(text: &str) -> Result<Self, ParseSquareError> {
        match text.as_bytes() {
            [file @ b'a'..=b'h', rank @ b'1'..=b'8'] => Ok(Self::new(file - b'a', rank - b'1')),
            _ => Err(ParseSquareError),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Move(u16);

impl Move {
    pub const NULL: Self = Self(0);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn from_square(square: Square) -> Self {
        Self(1u64 << square.0)
    }

    pub const fn contains(self, square: Square) -> bool {
        self.0 & (1u64 << square.0) != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    pub const fn lowest_square(self) -> Square {
        debug_assert!(!self.is_empty());
        Square(self.0.trailing_zeros() as u8)
    }
}

impl std::ops::BitAnd for Bitboard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl std::ops::BitOr for Bitboard {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::Not for Bitboard {
    type Output = Self;

    fn not(self) -> Self {
        Self(!self.0)
    }
}

impl std::ops::BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl std::ops::BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl std::fmt::Display for Bitboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for rank in (0..RANKS).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in 0..FILES {
                let symbol = if self.contains(Square::new(file, rank)) {
                    '1'
                } else {
                    '.'
                };
                write!(f, " {symbol}")?;
            }
            writeln!(f)?;
        }
        write!(f, "   a b c d e f g h")
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EverySide<T>([T; SIDES]);

impl<T: Copy> EverySide<T> {
    pub const fn new(value: T) -> Self {
        Self([value; SIDES])
    }
}

impl<T> std::ops::Index<Color> for EverySide<T> {
    type Output = T;

    fn index(&self, side: Color) -> &T {
        &self.0[side.index()]
    }
}

impl<T> std::ops::IndexMut<Color> for EverySide<T> {
    fn index_mut(&mut self, side: Color) -> &mut T {
        &mut self.0[side.index()]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EveryPiece<T>([T; PIECE_TYPES]);

impl<T: Copy> EveryPiece<T> {
    pub const fn new(value: T) -> Self {
        Self([value; PIECE_TYPES])
    }
}

impl<T> std::ops::Index<Piece> for EveryPiece<T> {
    type Output = T;

    fn index(&self, piece: Piece) -> &T {
        &self.0[piece.index()]
    }
}

impl<T> std::ops::IndexMut<Piece> for EveryPiece<T> {
    fn index_mut(&mut self, piece: Piece) -> &mut T {
        &mut self.0[piece.index()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EverySquare<T>([T; SQUARES]);

impl<T: Copy> EverySquare<T> {
    pub const fn new(value: T) -> Self {
        Self([value; SQUARES])
    }
}

impl<T> std::ops::Index<Square> for EverySquare<T> {
    type Output = T;

    fn index(&self, square: Square) -> &T {
        &self.0[square.index() & (SQUARES - 1)]
    }
}

impl<T> std::ops::IndexMut<Square> for EverySquare<T> {
    fn index_mut(&mut self, square: Square) -> &mut T {
        &mut self.0[square.index()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const ALL: [Self; SIDES] = [Self::White, Self::Black];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn opponent(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece {
    King = 0,
    Queen = 1,
    Rook = 2,
    Bishop = 3,
    Knight = 4,
    Pawn = 5,
}

impl Piece {
    pub const ALL: [Self; PIECE_TYPES] = [
        Self::King,
        Self::Queen,
        Self::Rook,
        Self::Bishop,
        Self::Knight,
        Self::Pawn,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn from_symbol(symbol: char) -> Option<(Color, Self)> {
        let side = if symbol.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let piece = match symbol.to_ascii_lowercase() {
            'k' => Self::King,
            'q' => Self::Queen,
            'r' => Self::Rook,
            'b' => Self::Bishop,
            'n' => Self::Knight,
            'p' => Self::Pawn,
            _ => return None,
        };
        Some((side, piece))
    }

    pub const fn symbol(self, side: Color) -> char {
        let symbol = match self {
            Self::King => 'k',
            Self::Queen => 'q',
            Self::Rook => 'r',
            Self::Bishop => 'b',
            Self::Knight => 'n',
            Self::Pawn => 'p',
        };
        match side {
            Color::White => symbol.to_ascii_uppercase(),
            Color::Black => symbol,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CastlingRights(u8);

impl CastlingRights {
    pub const WHITE_KING: Self = Self(1);
    pub const WHITE_QUEEN: Self = Self(2);
    pub const BLACK_KING: Self = Self(4);
    pub const BLACK_QUEEN: Self = Self(8);
    pub const NONE: Self = Self(0);
    pub const ALL: Self =
        Self(Self::WHITE_KING.0 | Self::WHITE_QUEEN.0 | Self::BLACK_KING.0 | Self::BLACK_QUEEN.0);

    pub const fn has(self, rights: Self) -> bool {
        self.0 & rights.0 != 0
    }

    pub fn insert(&mut self, rights: Self) {
        self.0 |= rights.0;
    }

    pub fn remove(&mut self, rights: Self) {
        self.0 &= !rights.0;
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for CastlingRights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == Self::NONE {
            return write!(f, "-");
        }
        let symbols = [
            (Self::WHITE_KING, 'K'),
            (Self::WHITE_QUEEN, 'Q'),
            (Self::BLACK_KING, 'k'),
            (Self::BLACK_QUEEN, 'q'),
        ];
        for (right, symbol) in symbols {
            if self.has(right) {
                write!(f, "{symbol}")?;
            }
        }
        Ok(())
    }
}
