use crate::board::Board;
use crate::types::{Bitboard, Color, Piece};

use super::{Score, Totals, collect_pawn_structure};

const PAWN_CACHE_BITS: u32 = 14;
const PAWN_KEY_MULTIPLIER: u64 = 0x9E37_79B9_7F4A_7C15;
const BLACK_PAWNS_ROTATION: u32 = 29;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PawnStructure {
    pub score: Score,
    pub passed: [Bitboard; 2],
}

#[derive(Clone, Copy, Default)]
struct Entry {
    pawns: [Bitboard; 2],
    structure: PawnStructure,
}

pub struct PawnCache {
    entries: Vec<Entry>,
}

impl Default for PawnCache {
    fn default() -> Self {
        Self {
            entries: vec![Entry::default(); 1 << PAWN_CACHE_BITS],
        }
    }
}

impl PawnCache {
    pub fn probe(&mut self, board: &Board) -> PawnStructure {
        let pawns = Color::ALL.map(|side| board.pieces_of(side, Piece::Pawn));
        let slot = &mut self.entries[slot_index(pawns)];
        if slot.pawns != pawns {
            let mut totals = Totals::default();
            let passed = Color::ALL.map(|side| collect_pawn_structure(board, side, &mut totals));
            *slot = Entry {
                pawns,
                structure: PawnStructure {
                    score: totals.0,
                    passed,
                },
            };
        }
        slot.structure
    }
}

fn slot_index([white, black]: [Bitboard; 2]) -> usize {
    let mixed =
        (white.0 ^ black.0.rotate_left(BLACK_PAWNS_ROTATION)).wrapping_mul(PAWN_KEY_MULTIPLIER);
    (mixed >> (u64::BITS - PAWN_CACHE_BITS)) as usize
}
