use std::sync::LazyLock;

use super::mapping::Mapping;
use crate::attacks::king_attacks;
use crate::types::{Bitboard, SQUARES, Square};

pub const MAX_PIECES: usize = 7;
const MAX_LEAD_PAWNS: usize = 5;
const PAWN_FILES: usize = 4;
const KING_PAIR_POSITIONS: u64 = 462;
const UNIQUE_TRIPLE_POSITIONS: u64 = 31_332;
const LEAD_TRIANGLE_SQUARES: usize = 10;
const LEAD_TRIANGLE_REGION: usize = 28;

const STM_FLAG: u8 = 1;
const MAPPED_FLAG: u8 = 2;
const WIN_PLIES_FLAG: u8 = 4;
const LOSS_PLIES_FLAG: u8 = 8;
const WIDE_FLAG: u8 = 16;
const SINGLE_VALUE_FLAG: u8 = 128;

const SPLIT_HEADER: u8 = 1;
const PAWNS_HEADER: u8 = 2;

const WDL_MAGIC: [u8; 4] = [0x71, 0xE8, 0x23, 0x5D];
const DTZ_MAGIC: [u8; 4] = [0xD7, 0x66, 0x0C, 0xA5];

const SPARSE_ENTRY_BYTES: usize = 6;
const TREE_NODE_BYTES: usize = 3;
const LEAF_MARKER: u16 = 0xFFF;
const DATA_ALIGNMENT: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Wdl,
    Dtz,
}

impl Kind {
    fn magic(self) -> [u8; 4] {
        match self {
            Self::Wdl => WDL_MAGIC,
            Self::Dtz => DTZ_MAGIC,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Material {
    pub piece_count: usize,
    pub has_pawns: bool,
    pub has_unique_pieces: bool,
    pub pawn_counts: [usize; 2],
    pub symmetric: bool,
}

pub enum Lookup {
    Value(i32),
    OtherSideToMove,
    Unavailable,
}

struct Indexing {
    map_pawns: [usize; SQUARES],
    map_b1h1h7: [usize; SQUARES],
    map_a1d1d4: [usize; SQUARES],
    map_kk: [[u64; SQUARES]; LEAD_TRIANGLE_SQUARES],
    binomial: [[u64; SQUARES]; MAX_PIECES],
    lead_pawn_index: [[u64; SQUARES]; MAX_LEAD_PAWNS + 1],
    lead_pawns_size: [[u64; PAWN_FILES]; MAX_LEAD_PAWNS + 1],
}

fn diagonal_offset(square: usize) -> i32 {
    (square / 8) as i32 - (square % 8) as i32
}

fn file_flipped(square: usize) -> usize {
    square ^ 7
}

fn rank_flipped(square: usize) -> usize {
    square ^ 56
}

fn diagonal_flipped(square: usize) -> usize {
    ((square >> 3) | (square << 3)) & 63
}

fn queenside_file(file: usize) -> usize {
    file.min(7 - file)
}

static INDEXING: LazyLock<Indexing> = LazyLock::new(|| {
    let mut map_b1h1h7 = [0; SQUARES];
    let mut code = 0;
    for (square, slot) in map_b1h1h7.iter_mut().enumerate() {
        if diagonal_offset(square) < 0 {
            *slot = code;
            code += 1;
        }
    }

    let mut map_a1d1d4 = [0; SQUARES];
    let mut diagonal = Vec::new();
    code = 0;
    for (square, slot) in map_a1d1d4.iter_mut().enumerate().take(LEAD_TRIANGLE_REGION) {
        let in_queenside = square % 8 <= 3;
        if diagonal_offset(square) < 0 && in_queenside {
            *slot = code;
            code += 1;
        } else if diagonal_offset(square) == 0 && in_queenside {
            diagonal.push(square);
        }
    }
    for square in diagonal {
        map_a1d1d4[square] = code;
        code += 1;
    }

    let mut map_kk = [[0; SQUARES]; LEAD_TRIANGLE_SQUARES];
    let mut both_on_diagonal = Vec::new();
    let mut code = 0;
    for (triangle_index, row) in map_kk.iter_mut().enumerate() {
        for (first, &first_triangle) in map_a1d1d4.iter().enumerate().take(LEAD_TRIANGLE_REGION) {
            let first_is_b1 = first == Square::B1.index();
            if first_triangle != triangle_index || (triangle_index == 0 && !first_is_b1) {
                continue;
            }
            let blocked = king_attacks(Square::from_index(first as u8))
                | Bitboard::from_square(Square::from_index(first as u8));
            for (second, slot) in row.iter_mut().enumerate() {
                if blocked.contains(Square::from_index(second as u8)) {
                    continue;
                }
                if diagonal_offset(first) == 0 && diagonal_offset(second) > 0 {
                    continue;
                }
                if diagonal_offset(first) == 0 && diagonal_offset(second) == 0 {
                    both_on_diagonal.push((triangle_index, second));
                } else {
                    *slot = code;
                    code += 1;
                }
            }
        }
    }
    for (triangle_index, second) in both_on_diagonal {
        map_kk[triangle_index][second] = code;
        code += 1;
    }

    let mut binomial = [[0; SQUARES]; MAX_PIECES];
    binomial[0][0] = 1;
    for squares in 1..SQUARES {
        for pieces in 0..MAX_PIECES.min(squares + 1) {
            let with_piece = if pieces > 0 {
                binomial[pieces - 1][squares - 1]
            } else {
                0
            };
            let without_piece = if pieces < squares {
                binomial[pieces][squares - 1]
            } else {
                0
            };
            binomial[pieces][squares] = with_piece + without_piece;
        }
    }

    let mut map_pawns = [0; SQUARES];
    let mut lead_pawn_index = [[0; SQUARES]; MAX_LEAD_PAWNS + 1];
    let mut lead_pawns_size = [[0; PAWN_FILES]; MAX_LEAD_PAWNS + 1];
    let mut available_squares: usize = 47;
    for lead_pawns in 1..=MAX_LEAD_PAWNS {
        for (file, size) in lead_pawns_size[lead_pawns].iter_mut().enumerate() {
            let mut index = 0;
            for rank in 1..7 {
                let square = rank * 8 + file;
                if lead_pawns == 1 {
                    map_pawns[square] = available_squares;
                    available_squares -= 1;
                    map_pawns[file_flipped(square)] = available_squares;
                    available_squares = available_squares.saturating_sub(1);
                }
                lead_pawn_index[lead_pawns][square] = index;
                index += binomial[lead_pawns - 1][map_pawns[square]];
            }
            *size = index;
        }
    }

    Indexing {
        map_pawns,
        map_b1h1h7,
        map_a1d1d4,
        map_kk,
        binomial,
        lead_pawn_index,
        lead_pawns_size,
    }
});

#[derive(Clone, Default)]
struct PairsData {
    flags: u8,
    min_symbol_length: u8,
    block_size: u64,
    span: u64,
    block_count: u64,
    block_length_count: u64,
    lowest_symbols: usize,
    tree: usize,
    sparse_index: usize,
    sparse_index_count: u64,
    block_lengths: usize,
    data: usize,
    base: Vec<u64>,
    symbol_lengths: Vec<u8>,
    pieces: [u8; MAX_PIECES],
    group_index: [u64; MAX_PIECES + 1],
    group_length: [usize; MAX_PIECES + 1],
    map_index: [usize; 4],
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl Reader<'_> {
    fn byte(&self, at: usize) -> Option<u8> {
        self.bytes.get(at).copied()
    }

    fn array<const N: usize>(&self, at: usize) -> Option<[u8; N]> {
        self.bytes.get(at..at.checked_add(N)?)?.try_into().ok()
    }

    fn u16_le(&self, at: usize) -> Option<u16> {
        self.array(at).map(u16::from_le_bytes)
    }

    fn u32_le(&self, at: usize) -> Option<u32> {
        self.array(at).map(u32::from_le_bytes)
    }

    fn u32_be_or_zero(&self, at: usize) -> u32 {
        self.array(at).map_or(0, u32::from_be_bytes)
    }

    fn u64_be(&self, at: usize) -> Option<u64> {
        self.array(at).map(u64::from_be_bytes)
    }

    fn tree_children(&self, tree: usize, symbol: u16) -> Option<(u16, u16)> {
        let node: [u8; TREE_NODE_BYTES] =
            self.array(tree + usize::from(symbol) * TREE_NODE_BYTES)?;
        let left = (u16::from(node[1] & 0xF) << 8) | u16::from(node[0]);
        let right = (u16::from(node[2]) << 4) | u16::from(node[1] >> 4);
        Some((left, right))
    }
}

fn symbol_length(
    reader: &Reader,
    data: &mut PairsData,
    symbol: u16,
    visited: &mut [bool],
) -> Option<u8> {
    *visited.get_mut(usize::from(symbol))? = true;
    let (left, right) = reader.tree_children(data.tree, symbol)?;
    if right == LEAF_MARKER {
        return Some(0);
    }
    for child in [left, right] {
        if !*visited.get(usize::from(child))? {
            let length = symbol_length(reader, data, child, visited)?;
            data.symbol_lengths[usize::from(child)] = length;
        }
    }
    let total = u16::from(data.symbol_lengths[usize::from(left)])
        + u16::from(data.symbol_lengths[usize::from(right)])
        + 1;
    u8::try_from(total).ok()
}

impl PairsData {
    fn read_sizes(&mut self, reader: &Reader, mut at: usize) -> Option<usize> {
        self.flags = reader.byte(at)?;
        at += 1;
        if self.flags & SINGLE_VALUE_FLAG != 0 {
            self.min_symbol_length = reader.byte(at)?;
            return Some(at + 1);
        }
        let groups = self.group_length.iter().position(|&length| length == 0)?;
        let table_size = self.group_index[groups];
        self.block_size = 1u64.checked_shl(u32::from(reader.byte(at)?))?;
        self.span = 1u64.checked_shl(u32::from(reader.byte(at + 1)?))?;
        self.sparse_index_count = table_size.div_ceil(self.span);
        let padding = u64::from(reader.byte(at + 2)?);
        self.block_count = u64::from(reader.u32_le(at + 3)?);
        self.block_length_count = self.block_count + padding;
        let max_symbol_length = reader.byte(at + 7)?;
        self.min_symbol_length = reader.byte(at + 8)?;
        at += 9;
        self.lowest_symbols = at;
        let lengths = usize::from(max_symbol_length.checked_sub(self.min_symbol_length)?) + 1;
        self.base = vec![0; lengths];
        for length in (0..lengths - 1).rev() {
            let lowest = u64::from(reader.u16_le(self.lowest_symbols + 2 * length)?);
            let next_lowest = u64::from(reader.u16_le(self.lowest_symbols + 2 * (length + 1))?);
            self.base[length] = self.base[length + 1]
                .wrapping_add(lowest)
                .wrapping_sub(next_lowest)
                / 2;
        }
        for (length, base) in self.base.iter_mut().enumerate() {
            let shift = 64 - length as u32 - u32::from(self.min_symbol_length);
            *base = base.checked_shl(shift).unwrap_or(0);
        }
        at += lengths * 2;
        let symbols = usize::from(reader.u16_le(at)?);
        at += 2;
        self.tree = at;
        self.symbol_lengths = vec![0; symbols];
        let mut visited = vec![false; symbols];
        for symbol in 0..symbols {
            if !visited[symbol] {
                let length = symbol_length(reader, self, symbol as u16, &mut visited)?;
                self.symbol_lengths[symbol] = length;
            }
        }
        Some(at + symbols * TREE_NODE_BYTES + (symbols & 1))
    }

    fn block_length(&self, reader: &Reader, block: u64) -> Option<i64> {
        if block >= self.block_length_count {
            return None;
        }
        let at = self.block_lengths + 2 * usize::try_from(block).ok()?;
        reader.u16_le(at).map(i64::from)
    }

    fn decompress(&self, reader: &Reader, index: u64) -> Option<u16> {
        if self.flags & SINGLE_VALUE_FLAG != 0 {
            return Some(u16::from(self.min_symbol_length));
        }
        let sparse = index / self.span;
        if sparse >= self.sparse_index_count {
            return None;
        }
        let entry = self.sparse_index + usize::try_from(sparse).ok()? * SPARSE_ENTRY_BYTES;
        let mut block = u64::from(reader.u32_le(entry)?);
        let mut offset = i64::from(reader.u16_le(entry + 4)?);
        offset += (index % self.span) as i64 - (self.span / 2) as i64;
        while offset < 0 {
            block = block.checked_sub(1)?;
            offset += self.block_length(reader, block)? + 1;
        }
        while offset > self.block_length(reader, block)? {
            offset -= self.block_length(reader, block)? + 1;
            block += 1;
        }

        let mut at = self.data + usize::try_from(block * self.block_size).ok()?;
        let mut buffer = reader.u64_be(at)?;
        at += 8;
        let mut buffer_bits = 64;
        let min_length = u32::from(self.min_symbol_length);
        let mut symbol;
        loop {
            let mut length = 0;
            while buffer < *self.base.get(length)? {
                length += 1;
            }
            let shift = 64 - length as u32 - min_length;
            symbol = ((buffer - self.base[length]) >> shift) as u16;
            symbol = symbol.wrapping_add(reader.u16_le(self.lowest_symbols + 2 * length)?);
            let expanded = i64::from(*self.symbol_lengths.get(usize::from(symbol))?) + 1;
            if offset < expanded {
                break;
            }
            offset -= expanded;
            let consumed = length as u32 + min_length;
            buffer = buffer.checked_shl(consumed).unwrap_or(0);
            buffer_bits -= consumed as i32;
            if buffer_bits <= 32 {
                buffer_bits += 32;
                buffer |= u64::from(reader.u32_be_or_zero(at)) << (64 - buffer_bits);
                at += 4;
            }
        }

        while self.symbol_lengths[usize::from(symbol)] != 0 {
            let (left, right) = reader.tree_children(self.tree, symbol)?;
            let left_expanded = i64::from(*self.symbol_lengths.get(usize::from(left))?) + 1;
            if offset < left_expanded {
                symbol = left;
            } else {
                offset -= left_expanded;
                symbol = right;
            }
        }
        reader
            .tree_children(self.tree, symbol)
            .map(|(left, _)| left)
    }
}

pub struct EncodedTable {
    mapping: Mapping,
    pairs: Vec<PairsData>,
    sides: usize,
    dtz_map: usize,
}

fn set_groups(
    material: &Material,
    data: &mut PairsData,
    order: [u8; 2],
    file: usize,
) -> Option<()> {
    let indexing = &*INDEXING;
    let mut groups = 0;
    let mut first_length: i32 = if material.has_pawns {
        0
    } else if material.has_unique_pieces {
        3
    } else {
        2
    };
    data.group_length[0] = 1;
    for piece in 1..material.piece_count {
        first_length -= 1;
        if first_length > 0 || data.pieces[piece] == data.pieces[piece - 1] {
            data.group_length[groups] += 1;
        } else {
            groups += 1;
            data.group_length[groups] = 1;
        }
    }
    groups += 1;
    data.group_length[groups] = 0;

    let both_sides_have_pawns = material.has_pawns && material.pawn_counts[1] > 0;
    let mut next = if both_sides_have_pawns { 2 } else { 1 };
    let mut free_squares = 64
        - data.group_length[0]
        - if both_sides_have_pawns {
            data.group_length[1]
        } else {
            0
        };
    let mut index: u64 = 1;
    let mut slot = 0;
    while next < groups || slot == usize::from(order[0]) || slot == usize::from(order[1]) {
        if slot == usize::from(order[0]) {
            data.group_index[0] = index;
            index = index.checked_mul(if material.has_pawns {
                *indexing
                    .lead_pawns_size
                    .get(data.group_length[0])?
                    .get(file)?
            } else if material.has_unique_pieces {
                UNIQUE_TRIPLE_POSITIONS
            } else {
                KING_PAIR_POSITIONS
            })?;
        } else if slot == usize::from(order[1]) {
            data.group_index[1] = index;
            index = index.checked_mul(
                *indexing
                    .binomial
                    .get(data.group_length[1])?
                    .get(48usize.checked_sub(data.group_length[0])?)?,
            )?;
        } else {
            data.group_index[next] = index;
            index = index.checked_mul(
                *indexing
                    .binomial
                    .get(data.group_length[next])?
                    .get(free_squares)?,
            )?;
            free_squares = free_squares.checked_sub(data.group_length[next])?;
            next += 1;
        }
        slot += 1;
        if slot > 16 {
            return None;
        }
    }
    data.group_index[groups] = index;
    Some(())
}

impl EncodedTable {
    pub fn parse(mapping: Mapping, kind: Kind, material: &Material) -> Option<Self> {
        let reader = Reader {
            bytes: mapping.bytes(),
        };
        if reader.array::<4>(0)? != kind.magic() {
            return None;
        }
        let header = reader.byte(4)?;
        if (header & PAWNS_HEADER != 0) != material.has_pawns {
            return None;
        }
        if kind == Kind::Wdl && (header & SPLIT_HEADER != 0) == material.symmetric {
            return None;
        }
        if material.piece_count > MAX_PIECES || material.piece_count < 3 {
            return None;
        }
        let sides = if kind == Kind::Wdl && !material.symmetric {
            2
        } else {
            1
        };
        let files = if material.has_pawns { PAWN_FILES } else { 1 };
        let both_sides_have_pawns = material.has_pawns && material.pawn_counts[1] > 0;
        let mut pairs = vec![PairsData::default(); 2 * PAWN_FILES];
        let slot = |side: usize, file: usize| side * PAWN_FILES + file;

        let mut at = 5;
        for file in 0..files {
            let first = reader.byte(at)?;
            let second = if both_sides_have_pawns {
                reader.byte(at + 1)?
            } else {
                0xFF
            };
            let orders = [[first & 0xF, second & 0xF], [first >> 4, second >> 4]];
            at += 1 + usize::from(both_sides_have_pawns);
            for piece in 0..material.piece_count {
                let byte = reader.byte(at)?;
                for side in 0..sides {
                    pairs[slot(side, file)].pieces[piece] =
                        if side == 1 { byte >> 4 } else { byte & 0xF };
                }
                at += 1;
            }
            for (side, order) in orders.into_iter().enumerate().take(sides) {
                set_groups(material, &mut pairs[slot(side, file)], order, file)?;
            }
        }
        at += at & 1;

        for file in 0..files {
            for side in 0..sides {
                at = pairs[slot(side, file)].read_sizes(&reader, at)?;
            }
        }

        let dtz_map = at;
        if kind == Kind::Dtz {
            for file in 0..files {
                let data = &mut pairs[slot(0, file)];
                if data.flags & MAPPED_FLAG == 0 {
                    continue;
                }
                if data.flags & WIDE_FLAG != 0 {
                    at += at & 1;
                    for map_index in &mut data.map_index {
                        *map_index = (at - dtz_map) / 2 + 1;
                        at += 2 * usize::from(reader.u16_le(at)?) + 2;
                    }
                } else {
                    for map_index in &mut data.map_index {
                        *map_index = at - dtz_map + 1;
                        at += usize::from(reader.byte(at)?) + 1;
                    }
                }
            }
            at += at & 1;
        }

        for file in 0..files {
            for side in 0..sides {
                let data = &mut pairs[slot(side, file)];
                data.sparse_index = at;
                at += usize::try_from(data.sparse_index_count).ok()? * SPARSE_ENTRY_BYTES;
            }
        }
        for file in 0..files {
            for side in 0..sides {
                let data = &mut pairs[slot(side, file)];
                data.block_lengths = at;
                at += usize::try_from(data.block_length_count).ok()? * 2;
            }
        }
        for file in 0..files {
            for side in 0..sides {
                let data = &mut pairs[slot(side, file)];
                at = at.next_multiple_of(DATA_ALIGNMENT);
                data.data = at;
                at += usize::try_from(data.block_count.saturating_mul(data.block_size)).ok()?;
            }
        }

        Some(Self {
            mapping,
            pairs,
            sides,
            dtz_map,
        })
    }

    fn pairs(&self, side: usize, file: usize) -> &PairsData {
        &self.pairs[(side % self.sides) * PAWN_FILES + file]
    }

    pub fn lookup(
        &self,
        material: &Material,
        kind: Kind,
        placement: &[(usize, u8)],
        side: usize,
        wdl: i32,
    ) -> Lookup {
        match self.try_lookup(material, kind, placement, side, wdl) {
            Some(lookup) => lookup,
            None => Lookup::Unavailable,
        }
    }

    fn try_lookup(
        &self,
        material: &Material,
        kind: Kind,
        placement: &[(usize, u8)],
        side: usize,
        wdl: i32,
    ) -> Option<Lookup> {
        let indexing = &*INDEXING;
        if placement.len() != material.piece_count {
            return None;
        }
        let mut squares = [0usize; MAX_PIECES];
        let mut pieces = [0u8; MAX_PIECES];
        let mut size = 0;
        let mut lead_pawns = 0;
        let mut file = 0;
        let lead_code = self.pairs(0, 0).pieces[0];
        if material.has_pawns {
            for &(square, code) in placement {
                if code == lead_code {
                    squares[size] = square;
                    pieces[size] = code;
                    size += 1;
                }
            }
            lead_pawns = size;
            if lead_pawns == 0 || lead_pawns > MAX_LEAD_PAWNS {
                return None;
            }
            let leading = (0..lead_pawns).max_by_key(|&i| indexing.map_pawns[squares[i]])?;
            squares.swap(0, leading);
            file = queenside_file(squares[0] % 8);
        }

        if kind == Kind::Dtz {
            let flags = self.pairs(0, file).flags;
            let stored_side = usize::from(flags & STM_FLAG);
            if stored_side != side && !(material.symmetric && !material.has_pawns) {
                return Some(Lookup::OtherSideToMove);
            }
        }

        for &(square, code) in placement {
            if material.has_pawns && code == lead_code {
                continue;
            }
            squares[size] = square;
            pieces[size] = code;
            size += 1;
        }

        let data = self.pairs(side, file);
        for i in lead_pawns..size.saturating_sub(1) {
            for j in i + 1..size {
                if data.pieces[i] == pieces[j] {
                    pieces.swap(i, j);
                    squares.swap(i, j);
                    break;
                }
            }
        }

        if squares[0] % 8 > 3 {
            for square in &mut squares[..size] {
                *square = file_flipped(*square);
            }
        }

        let mut index: u64;
        if material.has_pawns {
            index = *indexing.lead_pawn_index[lead_pawns].get(squares[0])?;
            squares[1..lead_pawns].sort_by_key(|&square| indexing.map_pawns[square]);
            for (i, &square) in squares.iter().enumerate().take(lead_pawns).skip(1) {
                index += indexing.binomial[i][indexing.map_pawns[square]];
            }
        } else {
            if squares[0] / 8 > 3 {
                for square in &mut squares[..size] {
                    *square = rank_flipped(*square);
                }
            }
            for i in 0..data.group_length[0] {
                let offset = diagonal_offset(squares[i]);
                if offset == 0 {
                    continue;
                }
                if offset > 0 {
                    for square in &mut squares[i..size] {
                        *square = diagonal_flipped(*square);
                    }
                }
                break;
            }
            index = if material.has_unique_pieces {
                unique_triple_index(indexing, &squares)
            } else {
                *indexing
                    .map_kk
                    .get(indexing.map_a1d1d4[squares[0]])?
                    .get(squares[1])?
            };
        }

        index = index.checked_mul(data.group_index[0])?;
        let mut group_start = data.group_length[0];
        let mut remaining_pawns = material.has_pawns && material.pawn_counts[1] > 0;
        let mut next = 1;
        while data.group_length[next] != 0 {
            let group_end = group_start + data.group_length[next];
            squares[group_start..group_end].sort_unstable();
            let mut combination = 0;
            for i in 0..data.group_length[next] {
                let square = squares[group_start + i];
                let below = squares[..group_start]
                    .iter()
                    .filter(|&&earlier| square > earlier)
                    .count();
                let rank_skip = if remaining_pawns { 8 } else { 0 };
                let mapped = square.checked_sub(below + rank_skip)?;
                combination += indexing.binomial.get(i + 1)?.get(mapped)?;
            }
            remaining_pawns = false;
            index += combination.checked_mul(data.group_index[next])?;
            group_start = group_end;
            next += 1;
        }

        let reader = Reader {
            bytes: self.mapping.bytes(),
        };
        let value = i32::from(data.decompress(&reader, index)?);
        Some(Lookup::Value(match kind {
            Kind::Wdl => value - 2,
            Kind::Dtz => self.dtz_plies(&reader, file, value, wdl)?,
        }))
    }

    fn dtz_plies(&self, reader: &Reader, file: usize, value: i32, wdl: i32) -> Option<i32> {
        const MAP_SLOT_BY_WDL: [usize; 5] = [1, 3, 0, 2, 0];
        let data = self.pairs(0, file);
        let flags = data.flags;
        let mut value = value;
        if flags & MAPPED_FLAG != 0 {
            let slot = MAP_SLOT_BY_WDL[usize::try_from(wdl + 2).ok()?.min(4)];
            let position = data.map_index[slot] + usize::try_from(value).ok()?;
            value = if flags & WIDE_FLAG != 0 {
                i32::from(reader.u16_le(self.dtz_map + 2 * position)?)
            } else {
                i32::from(reader.byte(self.dtz_map + position)?)
            };
        }
        let in_moves = (wdl == 2 && flags & WIN_PLIES_FLAG == 0)
            || (wdl == -2 && flags & LOSS_PLIES_FLAG == 0)
            || wdl == 1
            || wdl == -1;
        if in_moves {
            value *= 2;
        }
        Some(value + 1)
    }
}

fn unique_triple_index(indexing: &Indexing, squares: &[usize; MAX_PIECES]) -> u64 {
    let [first, second, third] = [squares[0], squares[1], squares[2]];
    let adjust_second = u64::from(second > first);
    let adjust_third = u64::from(third > first) + u64::from(third > second);
    let rank = |square: usize| (square / 8) as u64;
    if diagonal_offset(first) != 0 {
        (indexing.map_a1d1d4[first] as u64 * 63 + second as u64 - adjust_second) * 62 + third as u64
            - adjust_third
    } else if diagonal_offset(second) != 0 {
        (6 * 63 + rank(first) * 28 + indexing.map_b1h1h7[second] as u64) * 62 + third as u64
            - adjust_third
    } else if diagonal_offset(third) != 0 {
        6 * 63 * 62
            + 4 * 28 * 62
            + rank(first) * 7 * 28
            + (rank(second) - adjust_second) * 28
            + indexing.map_b1h1h7[third] as u64
    } else {
        6 * 63 * 62
            + 4 * 28 * 62
            + 4 * 7 * 28
            + rank(first) * 7 * 6
            + (rank(second) - adjust_second) * 6
            + (rank(third) - adjust_third)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type SquareMap = fn(usize) -> usize;

    #[test]
    fn index_tables_cover_their_ranges() {
        let indexing = &*INDEXING;
        let largest_king_pair = indexing.map_kk.iter().flatten().max().copied();
        let cases = [
            (
                "b1-h1-h7 triangle",
                indexing.map_b1h1h7.iter().max().copied().map(|v| v as u64),
                27,
            ),
            (
                "a1-d1-d4 triangle",
                indexing.map_a1d1d4.iter().max().copied().map(|v| v as u64),
                9,
            ),
            ("king pairs", largest_king_pair, KING_PAIR_POSITIONS - 1),
            (
                "pawn squares",
                indexing.map_pawns.iter().max().copied().map(|v| v as u64),
                47,
            ),
            ("choose 2 of 48", Some(indexing.binomial[2][48]), 1128),
            ("choose 5 of 63", Some(indexing.binomial[5][63]), 7_028_847),
            (
                "one lead pawn on file a",
                Some(indexing.lead_pawns_size[1][0]),
                6,
            ),
            (
                "two lead pawns on file a",
                Some(indexing.lead_pawns_size[2][0]),
                47 + 45 + 43 + 41 + 39 + 37,
            ),
        ];

        for (name, actual, expected) in cases {
            assert_eq!(actual, Some(expected), "{name}");
        }
    }

    #[test]
    fn square_symmetries_are_involutions() {
        let cases: [(&str, SquareMap); 3] = [
            ("file flip", file_flipped),
            ("rank flip", rank_flipped),
            ("diagonal flip", diagonal_flipped),
        ];

        for (name, flip) in cases {
            for square in 0..SQUARES {
                assert_eq!(flip(flip(square)), square, "{name}, square {square}");
            }
        }
    }
}
