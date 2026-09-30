pub const SIDES: usize = 2;
pub const PIECE_TYPES: usize = 6;
pub const SQUARES: usize = 64;
pub const FILES: u8 = 8;
pub const RANKS: u8 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MoveKind {
    Quiet = 0,
    DoublePush = 1,
    KingCastle = 2,
    QueenCastle = 3,
    Capture = 4,
    EnPassant = 5,
    KnightPromotion = 8,
    BishopPromotion = 9,
    RookPromotion = 10,
    QueenPromotion = 11,
    KnightPromotionCapture = 12,
    BishopPromotionCapture = 13,
    RookPromotionCapture = 14,
    QueenPromotionCapture = 15,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Move(u16);

impl Move {
    pub const NULL: Self = Self(0);

    pub const fn new(from: Square, to: Square, kind: MoveKind) -> Self {
        Self(from.0 as u16 | (to.0 as u16) << 6 | (kind as u16) << 12)
    }

    pub const fn from(self) -> Square {
        Square((self.0 & 0x3F) as u8)
    }

    pub const fn to(self) -> Square {
        Square((self.0 >> 6 & 0x3F) as u8)
    }

    pub const fn flags(self) -> u8 {
        (self.0 >> 12) as u8
    }

    pub const fn is_capture(self) -> bool {
        self.flags() & 4 != 0
    }

    pub const fn is_promotion(self) -> bool {
        self.flags() & 8 != 0
    }

    pub const fn kind(self) -> MoveKind {
        match self.flags() {
            0 => MoveKind::Quiet,
            1 => MoveKind::DoublePush,
            2 => MoveKind::KingCastle,
            3 => MoveKind::QueenCastle,
            4 => MoveKind::Capture,
            5 => MoveKind::EnPassant,
            8 => MoveKind::KnightPromotion,
            9 => MoveKind::BishopPromotion,
            10 => MoveKind::RookPromotion,
            11 => MoveKind::QueenPromotion,
            12 => MoveKind::KnightPromotionCapture,
            13 => MoveKind::BishopPromotionCapture,
            14 => MoveKind::RookPromotionCapture,
            _ => MoveKind::QueenPromotionCapture,
        }
    }

    pub const fn promotion(self) -> Option<Piece> {
        if !self.is_promotion() {
            return None;
        }
        let pieces = [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen];
        Some(pieces[(self.flags() & 3) as usize])
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == Self::NULL {
            return write!(f, "0000");
        }
        write!(f, "{}{}", self.from(), self.to())?;
        if self.is_promotion() {
            let symbol = ['n', 'b', 'r', 'q'][(self.flags() & 3) as usize];
            write!(f, "{symbol}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    pub const fn rank(rank: u8) -> Self {
        Self(0xFF << (rank * 8))
    }

    pub const fn file(file: u8) -> Self {
        Self(0x0101_0101_0101_0101 << file)
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

pub struct Squares(u64);

impl Iterator for Squares {
    type Item = Square;

    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            return None;
        }
        let square = Square(self.0.trailing_zeros() as u8);
        self.0 &= self.0 - 1;
        Some(square)
    }
}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = Squares;

    fn into_iter(self) -> Squares {
        Squares(self.0)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn after_move(self, from: Square, to: Square) -> Self {
        Self(
            self.0
                & CASTLING_KEPT_AFTER_TOUCH[from.index()]
                & CASTLING_KEPT_AFTER_TOUCH[to.index()],
        )
    }
}

const CASTLING_KEPT_AFTER_TOUCH: [u8; SQUARES] = {
    let mut kept = [CastlingRights::ALL.0; SQUARES];
    kept[0] = !CastlingRights::WHITE_QUEEN.0;
    kept[4] = !(CastlingRights::WHITE_KING.0 | CastlingRights::WHITE_QUEEN.0);
    kept[7] = !CastlingRights::WHITE_KING.0;
    kept[56] = !CastlingRights::BLACK_QUEEN.0;
    kept[60] = !(CastlingRights::BLACK_KING.0 | CastlingRights::BLACK_QUEEN.0);
    kept[63] = !CastlingRights::BLACK_KING.0;
    kept
};

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
