mod fen;
mod makemove;
mod movegen;
mod perft;

pub use fen::{FenError, KIWIPETE, START_POSITION};
pub use movegen::{MAX_MOVES, MoveList};

use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};
use crate::types::{
    Bitboard, CastlingRights, Color, EveryPiece, EverySide, EverySquare, FILES, Move, Piece, RANKS,
    Square,
};

pub const MAX_GAME_MOVES: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameState {
    pub active_color: Color,
    pub castling: CastlingRights,
    pub en_passant: Option<Square>,
    pub halfmove_clock: u8,
    pub fullmove_number: u16,
    pub zobrist_key: u64,
    pub captured: Option<Piece>,
    pub played_move: Move,
}

impl GameState {
    pub const fn new() -> Self {
        Self {
            active_color: Color::White,
            castling: CastlingRights::NONE,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            zobrist_key: 0,
            captured: None,
            played_move: Move::NULL,
        }
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct History {
    list: [GameState; MAX_GAME_MOVES],
    count: usize,
}

impl History {
    pub fn new() -> Self {
        Self {
            list: [GameState::new(); MAX_GAME_MOVES],
            count: 0,
        }
    }

    pub fn push(&mut self, state: GameState) {
        self.list[self.count] = state;
        self.count += 1;
    }

    pub fn pop(&mut self) -> Option<GameState> {
        self.count = self.count.checked_sub(1)?;
        Some(self.list[self.count])
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn as_slice(&self) -> &[GameState] {
        &self.list[..self.count]
    }
}

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for History {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for History {}

impl std::fmt::Debug for History {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    pieces: EveryPiece<Bitboard>,
    colors: EverySide<Bitboard>,
    mailbox: EverySquare<Option<Piece>>,
    pub state: GameState,
    pub history: History,
}

impl Board {
    pub fn new() -> Self {
        Self {
            pieces: EveryPiece::new(Bitboard::empty()),
            colors: EverySide::new(Bitboard::empty()),
            mailbox: EverySquare::new(None),
            state: GameState::new(),
            history: History::new(),
        }
    }

    pub fn pieces(&self, piece: Piece) -> Bitboard {
        self.pieces[piece]
    }

    pub fn occupied_by(&self, side: Color) -> Bitboard {
        self.colors[side]
    }

    pub fn pieces_of(&self, side: Color, piece: Piece) -> Bitboard {
        self.pieces[piece] & self.colors[side]
    }

    pub fn occupancy(&self) -> Bitboard {
        self.colors[Color::White] | self.colors[Color::Black]
    }

    pub fn diagonal_sliders(&self, side: Color) -> Bitboard {
        (self.pieces[Piece::Bishop] | self.pieces[Piece::Queen]) & self.colors[side]
    }

    pub fn orthogonal_sliders(&self, side: Color) -> Bitboard {
        (self.pieces[Piece::Rook] | self.pieces[Piece::Queen]) & self.colors[side]
    }

    pub fn piece_on(&self, square: Square) -> Option<Piece> {
        self.mailbox[square]
    }

    pub fn colored_piece_on(&self, square: Square) -> Option<(Color, Piece)> {
        let piece = self.mailbox[square]?;
        let side = if self.colors[Color::White].contains(square) {
            Color::White
        } else {
            Color::Black
        };
        Some((side, piece))
    }

    pub fn king_square(&self, side: Color) -> Square {
        let kings = self.pieces_of(side, Piece::King);
        debug_assert_eq!(kings.count(), 1);
        Square(kings.0.trailing_zeros() as u8)
    }

    pub fn attackers_to(&self, square: Square, occupancy: Bitboard) -> Bitboard {
        let pawns = (pawn_attacks(Color::White, square)
            & self.pieces_of(Color::Black, Piece::Pawn))
            | (pawn_attacks(Color::Black, square) & self.pieces_of(Color::White, Piece::Pawn));
        let diagonal = self.pieces[Piece::Bishop] | self.pieces[Piece::Queen];
        let orthogonal = self.pieces[Piece::Rook] | self.pieces[Piece::Queen];
        pawns
            | (knight_attacks(square) & self.pieces[Piece::Knight])
            | (king_attacks(square) & self.pieces[Piece::King])
            | (bishop_attacks(square, occupancy) & diagonal)
            | (rook_attacks(square, occupancy) & orthogonal)
    }

    pub fn is_attacked(&self, square: Square, by: Color, occupancy: Bitboard) -> bool {
        !(self.attackers_to(square, occupancy) & self.colors[by]).is_empty()
    }

    fn put_piece(&mut self, side: Color, piece: Piece, square: Square) {
        debug_assert!(self.mailbox[square].is_none());
        let bit = Bitboard::from_square(square);
        self.pieces[piece] |= bit;
        self.colors[side] |= bit;
        self.mailbox[square] = Some(piece);
    }

    fn remove_piece(&mut self, side: Color, piece: Piece, square: Square) {
        debug_assert_eq!(self.mailbox[square], Some(piece));
        debug_assert!(self.colors[side].contains(square));
        let mask = !Bitboard::from_square(square);
        self.pieces[piece] &= mask;
        self.colors[side] &= mask;
        self.mailbox[square] = None;
    }

    fn move_piece(&mut self, side: Color, piece: Piece, from: Square, to: Square) {
        debug_assert_eq!(self.mailbox[from], Some(piece));
        debug_assert!(self.colors[side].contains(from));
        debug_assert!(self.mailbox[to].is_none());
        let bits = Bitboard::from_square(from) | Bitboard::from_square(to);
        self.pieces[piece] ^= bits;
        self.colors[side] ^= bits;
        self.mailbox[from] = None;
        self.mailbox[to] = Some(piece);
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for rank in (0..RANKS).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in 0..FILES {
                let square = Square::new(file, rank);
                let symbol = self
                    .colored_piece_on(square)
                    .map_or('.', |(side, piece)| piece.symbol(side));
                write!(f, " {symbol}")?;
            }
            writeln!(f)?;
        }
        write!(f, "   a b c d e f g h")
    }
}
