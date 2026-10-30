mod encoding;
mod mapping;

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::OnceLock;

use encoding::{EncodedTable, Kind, Lookup, MAX_PIECES, Material};
use mapping::Mapping;

use crate::board::{Board, GameState};
use crate::types::{CastlingRights, Color, Move, Piece};

const WDL_EXTENSION: &str = "rtbw";
const DTZ_EXTENSION: &str = "rtbz";
const MAX_DTZ: i32 = 1 << 18;
const FIFTY_MOVE_PLIES: i32 = 100;
const COUNTED_PIECES: [Piece; 5] = [
    Piece::Queen,
    Piece::Rook,
    Piece::Bishop,
    Piece::Knight,
    Piece::Pawn,
];
const COUNT_BITS: u32 = 3;
const BLACK_COLOR_CODE: u8 = 8;
const FLIPPED_RANKS: usize = 56;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Wdl {
    Loss = -2,
    BlessedLoss = -1,
    Draw = 0,
    CursedWin = 1,
    Win = 2,
}

impl Wdl {
    fn from_value(value: i32) -> Option<Self> {
        match value {
            -2 => Some(Self::Loss),
            -1 => Some(Self::BlessedLoss),
            0 => Some(Self::Draw),
            1 => Some(Self::CursedWin),
            2 => Some(Self::Win),
            _ => None,
        }
    }

    pub const fn value(self) -> i32 {
        self as i32
    }

    pub fn negated(self) -> Self {
        Self::from_value(-self.value()).expect("negated wdl is a wdl")
    }

    fn dtz_before_zeroing(self) -> i32 {
        match self {
            Self::Win => 1,
            Self::CursedWin => FIFTY_MOVE_PLIES + 1,
            Self::Draw => 0,
            Self::BlessedLoss => -(FIFTY_MOVE_PLIES + 1),
            Self::Loss => -1,
        }
    }

    fn root_rank(self) -> i32 {
        match self {
            Self::Loss => -MAX_DTZ,
            Self::BlessedLoss => -MAX_DTZ + FIFTY_MOVE_PLIES + 1,
            Self::Draw => 0,
            Self::CursedWin => MAX_DTZ - FIFTY_MOVE_PLIES - 1,
            Self::Win => MAX_DTZ,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootMoves {
    pub moves: Vec<Move>,
    pub wdl: Wdl,
}

type MaterialKey = u32;

fn material_key(counts: [[u8; COUNTED_PIECES.len()]; 2]) -> MaterialKey {
    counts.iter().flatten().fold(0, |key, &count| {
        (key << COUNT_BITS) | MaterialKey::from(count)
    })
}

fn board_counts(board: &Board) -> [[u8; COUNTED_PIECES.len()]; 2] {
    Color::ALL.map(|side| COUNTED_PIECES.map(|piece| board.pieces_of(side, piece).count() as u8))
}

fn side_counts(side: &str) -> Option<[u8; COUNTED_PIECES.len()]> {
    let mut kings = 0;
    let mut counts = [0u8; COUNTED_PIECES.len()];
    for symbol in side.chars() {
        let (_, piece) = Piece::from_symbol(symbol)?;
        if piece == Piece::King {
            kings += 1;
            continue;
        }
        let slot = COUNTED_PIECES
            .iter()
            .position(|&counted| counted == piece)?;
        counts[slot] += 1;
    }
    (kings == 1).then_some(counts)
}

fn piece_code(side: Color, piece: Piece) -> u8 {
    let kind = match piece {
        Piece::Pawn => 1,
        Piece::Knight => 2,
        Piece::Bishop => 3,
        Piece::Rook => 4,
        Piece::Queen => 5,
        Piece::King => 6,
    };
    match side {
        Color::White => kind,
        Color::Black => kind | BLACK_COLOR_CODE,
    }
}

struct Table {
    material: Material,
    key: MaterialKey,
    wdl_path: PathBuf,
    dtz_path: Option<PathBuf>,
    wdl: OnceLock<Option<EncodedTable>>,
    dtz: OnceLock<Option<EncodedTable>>,
}

impl Table {
    fn new(
        name: &str,
        wdl_path: PathBuf,
        dtz_path: Option<PathBuf>,
    ) -> Option<(Self, MaterialKey)> {
        let (stronger, weaker) = name.split_once('v')?;
        let stronger_counts = side_counts(stronger)?;
        let weaker_counts = side_counts(weaker)?;
        let piece_count = stronger.len() + weaker.len();
        if !(3..=MAX_PIECES).contains(&piece_count) {
            return None;
        }
        let pawn_slot = COUNTED_PIECES.len() - 1;
        let stronger_pawns = usize::from(stronger_counts[pawn_slot]);
        let weaker_pawns = usize::from(weaker_counts[pawn_slot]);
        let stronger_leads =
            weaker_pawns == 0 || (stronger_pawns > 0 && weaker_pawns >= stronger_pawns);
        let pawn_counts = if stronger_leads {
            [stronger_pawns, weaker_pawns]
        } else {
            [weaker_pawns, stronger_pawns]
        };
        let has_unique_pieces = [stronger_counts, weaker_counts]
            .iter()
            .any(|counts| counts.contains(&1));
        let key = material_key([stronger_counts, weaker_counts]);
        let mirrored_key = material_key([weaker_counts, stronger_counts]);
        let table = Self {
            material: Material {
                piece_count,
                has_pawns: stronger_pawns + weaker_pawns > 0,
                has_unique_pieces,
                pawn_counts,
                symmetric: key == mirrored_key,
            },
            key,
            wdl_path,
            dtz_path,
            wdl: OnceLock::new(),
            dtz: OnceLock::new(),
        };
        Some((table, mirrored_key))
    }

    fn encoded(&self, kind: Kind) -> Option<&EncodedTable> {
        let (cell, path) = match kind {
            Kind::Wdl => (&self.wdl, Some(&self.wdl_path)),
            Kind::Dtz => (&self.dtz, self.dtz_path.as_ref()),
        };
        cell.get_or_init(|| {
            let mapping = Mapping::open(path?).ok()?;
            EncodedTable::parse(mapping, kind, &self.material)
        })
        .as_ref()
    }
}

pub struct Tablebases {
    tables: Vec<Table>,
    by_key: HashMap<MaterialKey, usize>,
    max_pieces: usize,
}

fn table_files(directories: &[PathBuf], extension: &str) -> HashMap<String, PathBuf> {
    let mut found = HashMap::new();
    for directory in directories {
        let Ok(entries) = fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension() != Some(OsStr::new(extension)) {
                continue;
            }
            if let Some(name) = path.file_stem().and_then(OsStr::to_str) {
                found.entry(name.to_owned()).or_insert(path);
            }
        }
    }
    found
}

impl Tablebases {
    pub fn open(paths: &str) -> io::Result<Self> {
        let directories: Vec<PathBuf> = std::env::split_paths(paths).collect();
        if let Some(missing) = directories.iter().find(|directory| !directory.is_dir()) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("{} is not a directory", missing.display()),
            ));
        }
        let mut dtz_files = table_files(&directories, DTZ_EXTENSION);
        let mut wdl_files: Vec<(String, PathBuf)> = table_files(&directories, WDL_EXTENSION)
            .into_iter()
            .collect();
        wdl_files.sort();

        let mut tablebases = Self {
            tables: Vec::new(),
            by_key: HashMap::new(),
            max_pieces: 0,
        };
        for (name, wdl_path) in wdl_files {
            let dtz_path = dtz_files.remove(&name);
            let Some((table, mirrored_key)) = Table::new(&name, wdl_path, dtz_path) else {
                continue;
            };
            let index = tablebases.tables.len();
            tablebases.max_pieces = tablebases.max_pieces.max(table.material.piece_count);
            tablebases.by_key.insert(table.key, index);
            tablebases.by_key.insert(mirrored_key, index);
            tablebases.tables.push(table);
        }
        Ok(tablebases)
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    pub fn max_pieces(&self) -> usize {
        self.max_pieces
    }

    pub fn covers(&self, board: &Board) -> bool {
        board.state.castling == CastlingRights::NONE
            && board.occupancy().count() as usize <= self.max_pieces
    }

    fn probe_table(&self, board: &Board, kind: Kind, wdl: Wdl) -> Lookup {
        if board.occupancy().count() == 2 {
            return Lookup::Value(0);
        }
        let key = material_key(board_counts(board));
        let Some(table) = self.by_key.get(&key).map(|&index| &self.tables[index]) else {
            return Lookup::Unavailable;
        };
        let Some(encoded) = table.encoded(kind) else {
            return Lookup::Unavailable;
        };
        let black_to_move = board.state.active_color == Color::Black;
        let flipped = (table.material.symmetric && black_to_move) || key != table.key;
        let (square_flip, color_flip) = if flipped {
            (FLIPPED_RANKS, BLACK_COLOR_CODE)
        } else {
            (0, 0)
        };
        let mut placement = [(0, 0); MAX_PIECES];
        let mut size = 0;
        for square in board.occupancy() {
            if size == MAX_PIECES {
                return Lookup::Unavailable;
            }
            let (side, piece) = board.colored_piece_on(square).expect("occupied square");
            placement[size] = (
                square.index() ^ square_flip,
                piece_code(side, piece) ^ color_flip,
            );
            size += 1;
        }
        let side = usize::from(flipped) ^ usize::from(black_to_move);
        encoded.lookup(&table.material, kind, &placement[..size], side, wdl.value())
    }

    fn table_wdl(&self, board: &Board) -> Option<Wdl> {
        match self.probe_table(board, Kind::Wdl, Wdl::Draw) {
            Lookup::Value(value) => Wdl::from_value(value),
            Lookup::OtherSideToMove | Lookup::Unavailable => None,
        }
    }

    fn search_zeroing_moves(
        &self,
        board: &mut Board,
        pawn_moves_zero: bool,
    ) -> Option<(Wdl, bool)> {
        let moves = board.legal_moves();
        let mut best = Wdl::Loss;
        let mut searched = 0;
        for &candidate in &moves {
            let pawn_move = board.piece_on(candidate.from()) == Some(Piece::Pawn);
            if !candidate.is_capture() && !(pawn_moves_zero && pawn_move) {
                continue;
            }
            searched += 1;
            board.make_move(candidate);
            let value = self.search_zeroing_moves(board, false);
            board.unmake_move();
            let value = value?.0.negated();
            if value > best {
                best = value;
                if value == Wdl::Win {
                    return Some((value, true));
                }
            }
        }
        let every_move_searched = searched > 0 && searched == moves.len();
        let stored = if every_move_searched {
            best
        } else {
            self.table_wdl(board)?
        };
        if best >= stored {
            return Some((best, best > Wdl::Draw || every_move_searched));
        }
        Some((stored, false))
    }

    pub fn probe_wdl(&self, board: &mut Board) -> Option<Wdl> {
        if !self.covers(board) {
            return None;
        }
        self.search_zeroing_moves(board, false).map(|(wdl, _)| wdl)
    }

    pub fn probe_dtz(&self, board: &mut Board) -> Option<i32> {
        if !self.covers(board) {
            return None;
        }
        self.dtz(board)
    }

    fn dtz(&self, board: &mut Board) -> Option<i32> {
        let (wdl, zeroing_is_best) = self.search_zeroing_moves(board, true)?;
        if wdl == Wdl::Draw {
            return Some(0);
        }
        if zeroing_is_best {
            return Some(wdl.dtz_before_zeroing());
        }
        match self.probe_table(board, Kind::Dtz, wdl) {
            Lookup::Value(dtz) => {
                let fifty_move_shift = if matches!(wdl, Wdl::CursedWin | Wdl::BlessedLoss) {
                    FIFTY_MOVE_PLIES
                } else {
                    0
                };
                Some((dtz + fifty_move_shift) * wdl.value().signum())
            }
            Lookup::Unavailable => None,
            Lookup::OtherSideToMove => self.dtz_through_replies(board, wdl),
        }
    }

    fn dtz_through_replies(&self, board: &mut Board, wdl: Wdl) -> Option<i32> {
        let mut best = i32::MAX;
        for candidate in board.legal_moves() {
            let zeroing =
                candidate.is_capture() || board.piece_on(candidate.from()) == Some(Piece::Pawn);
            board.make_move(candidate);
            let reply_dtz = if zeroing {
                self.search_zeroing_moves(board, false)
                    .map(|(reply, _)| -reply.dtz_before_zeroing())
            } else {
                self.dtz(board).map(|dtz| -dtz)
            };
            let mates = reply_dtz == Some(1) && board.in_check() && board.legal_moves().is_empty();
            board.unmake_move();
            let mut dtz = reply_dtz?;
            if mates {
                best = 1;
            }
            if !zeroing {
                dtz += dtz.signum();
            }
            if dtz < best && dtz.signum() == wdl.value().signum() {
                best = dtz;
            }
        }
        Some(if best == i32::MAX { -1 } else { best })
    }

    pub fn root_moves(&self, board: &mut Board) -> Option<RootMoves> {
        if !self.covers(board) {
            return None;
        }
        let ranked = self
            .rank_by_dtz(board)
            .or_else(|| self.rank_by_wdl(board))?;
        let best_rank = ranked.iter().map(|&(_, rank, _)| rank).max()?;
        let wdl = ranked
            .iter()
            .filter(|&&(_, rank, _)| rank == best_rank)
            .map(|&(_, _, wdl)| wdl)
            .max()?;
        let moves = ranked
            .into_iter()
            .filter(|&(_, rank, _)| rank == best_rank)
            .map(|(candidate, _, _)| candidate)
            .collect();
        Some(RootMoves { moves, wdl })
    }

    fn rank_by_dtz(&self, board: &mut Board) -> Option<Vec<(Move, i32, Wdl)>> {
        let clock = i32::from(board.state.halfmove_clock);
        let repeated = repeated_since_zeroing(board);
        let mut ranked = Vec::new();
        for candidate in board.legal_moves() {
            board.make_move(candidate);
            let dtz = self.root_child_dtz(board);
            board.unmake_move();
            let dtz = dtz?;
            let rank = if dtz > 0 {
                if dtz + clock < FIFTY_MOVE_PLIES && !repeated {
                    MAX_DTZ
                } else {
                    MAX_DTZ - (dtz + clock)
                }
            } else if dtz < 0 {
                if -dtz * 2 + clock < FIFTY_MOVE_PLIES {
                    -MAX_DTZ
                } else {
                    -MAX_DTZ + (-dtz + clock)
                }
            } else {
                0
            };
            ranked.push((candidate, rank, wdl_of_root_dtz(dtz, clock)));
        }
        Some(ranked)
    }

    fn root_child_dtz(&self, board: &mut Board) -> Option<i32> {
        let mut dtz = if board.state.halfmove_clock == 0 {
            self.probe_wdl(board)?.negated().dtz_before_zeroing()
        } else if drawn_by_rule(board) {
            0
        } else {
            let dtz = -self.probe_dtz(board)?;
            dtz + dtz.signum()
        };
        if dtz == 2 && board.in_check() && board.legal_moves().is_empty() {
            dtz = 1;
        }
        Some(dtz)
    }

    fn rank_by_wdl(&self, board: &mut Board) -> Option<Vec<(Move, i32, Wdl)>> {
        let mut ranked = Vec::new();
        for candidate in board.legal_moves() {
            board.make_move(candidate);
            let wdl = if drawn_by_rule(board) {
                Some(Wdl::Draw)
            } else {
                self.probe_wdl(board).map(Wdl::negated)
            };
            board.unmake_move();
            let wdl = wdl?;
            ranked.push((candidate, wdl.root_rank(), wdl));
        }
        Some(ranked)
    }
}

fn wdl_of_root_dtz(dtz: i32, clock: i32) -> Wdl {
    match dtz.signum() {
        0 => Wdl::Draw,
        1 if dtz + clock <= FIFTY_MOVE_PLIES => Wdl::Win,
        1 => Wdl::CursedWin,
        _ if -dtz + clock <= FIFTY_MOVE_PLIES => Wdl::Loss,
        _ => Wdl::BlessedLoss,
    }
}

fn reversible_states(board: &Board) -> &[GameState] {
    let states = &board.history;
    let reversible = usize::from(board.state.halfmove_clock).min(states.len());
    &states[states.len() - reversible..]
}

fn repeated_since_zeroing(board: &Board) -> bool {
    let states = reversible_states(board);
    let current = board.state.zobrist_key;
    states.iter().enumerate().any(|(index, state)| {
        states[index + 1..]
            .iter()
            .map(|later| later.zobrist_key)
            .chain(std::iter::once(current))
            .any(|later| later == state.zobrist_key)
    })
}

fn drawn_by_rule(board: &mut Board) -> bool {
    let key = board.state.zobrist_key;
    let earlier_occurrences = reversible_states(board)
        .iter()
        .filter(|state| state.zobrist_key == key)
        .count();
    let fifty_moves = i32::from(board.state.halfmove_clock) >= FIFTY_MOVE_PLIES
        && !(board.in_check() && board.legal_moves().is_empty());
    earlier_occurrences >= 2 || fifty_moves
}

#[cfg(test)]
pub mod testing {
    use super::Tablebases;

    const LOCAL_TABLES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bin/syzygy");

    pub fn local_tablebases() -> Option<Tablebases> {
        let tablebases = Tablebases::open(LOCAL_TABLES)
            .ok()
            .filter(|tables| !tables.is_empty());
        if tablebases.is_none() {
            eprintln!("skipping: no Syzygy tables in {LOCAL_TABLES}");
        }
        tablebases
    }
}

#[cfg(test)]
mod tests {
    use super::testing::local_tablebases;
    use super::*;
    use crate::board::testing::{board, find_move};

    #[test]
    fn reads_material_from_table_names() {
        let cases = [
            ("KRvK", Some((3, false, true, [0, 0], false))),
            ("KNNvK", Some((4, false, false, [0, 0], false))),
            ("KRvKR", Some((4, false, true, [0, 0], true))),
            ("KPvK", Some((3, true, true, [1, 0], false))),
            ("KPPvKP", Some((5, true, true, [1, 2], false))),
            ("KRPvKP", Some((5, true, true, [1, 1], false))),
            ("KPPvKPP", Some((6, true, false, [2, 2], true))),
            ("KQRBNvK", Some((6, false, true, [0, 0], false))),
            ("KRK", None),
            ("KRvKvK", None),
            ("RvK", None),
            ("KKvK", None),
            ("KRvKX", None),
            ("KQRBNPvKP", None),
        ];

        for (name, expected) in cases {
            let table = Table::new(name, PathBuf::from(name), None).map(|(table, _)| {
                let material = table.material;
                (
                    material.piece_count,
                    material.has_pawns,
                    material.has_unique_pieces,
                    material.pawn_counts,
                    material.symmetric,
                )
            });
            assert_eq!(table, expected, "table {name}");
        }
    }

    #[test]
    fn probes_match_reference_values() {
        let Some(tablebases) = local_tablebases() else {
            return;
        };
        let cases = [
            ("kQ6/8/8/8/8/8/3K4/8 b - - 0 1", Wdl::Draw, 0),
            ("8/k7/8/R7/8/8/4K3/8 b - - 0 1", Wdl::Loss, -24),
            ("8/4K3/8/8/8/5P2/8/7k b - - 0 1", Wdl::Loss, -2),
            ("8/8/B7/5k2/8/8/8/4K1N1 w - - 0 1", Wdl::Win, 57),
            ("8/K2N4/6k1/8/3N4/8/8/8 b - - 0 1", Wdl::Draw, 0),
            ("6k1/8/1R6/K5q1/8/8/8/8 w - - 0 1", Wdl::Loss, -48),
            ("3k4/8/8/2B5/4r3/8/2K5/8 w - - 0 1", Wdl::Draw, 0),
            ("8/3R4/8/8/1k3n2/8/8/4K3 w - - 0 1", Wdl::Win, 3),
            ("8/8/3p4/8/8/6P1/2K5/5k2 b - - 0 1", Wdl::Draw, 0),
            ("8/8/8/5K2/8/k4q2/6Q1/8 w - - 0 1", Wdl::Win, 1),
            ("8/3k4/8/8/R7/8/1r6/2K5 w - - 0 1", Wdl::Win, 1),
            ("1B6/8/5k2/P7/8/8/8/7K w - - 0 1", Wdl::Win, 1),
            ("8/4r3/7R/8/1p6/8/7K/5k2 w - - 0 1", Wdl::Loss, -12),
            ("8/8/5p2/8/4k3/7K/2q5/4Q3 b - - 0 1", Wdl::Draw, 0),
            ("4K3/8/6P1/6p1/8/8/6P1/2k5 b - - 0 1", Wdl::Loss, -2),
            ("8/8/8/2N5/8/1k1p3K/8/8 b - - 0 1", Wdl::Draw, 0),
            ("8/8/7k/1R6/8/2K5/8/4Q3 b - - 0 1", Wdl::Loss, -5),
            ("3b4/P7/8/3K4/8/k7/8/8 w - - 0 1", Wdl::Win, 1),
            ("6B1/8/B7/2K1B3/8/8/8/1k6 w - - 0 1", Wdl::Win, 8),
            ("2K5/8/5B2/3k4/1B6/8/8/1N6 b - - 0 1", Wdl::Loss, -21),
            ("8/8/8/3k4/2n5/8/7B/4KB2 w - - 0 1", Wdl::CursedWin, 111),
            ("8/8/8/2N5/1K6/8/1k6/b2b4 b - - 0 1", Wdl::CursedWin, 113),
            ("2b2N1k/8/8/8/1K6/8/1b6/8 w - - 0 1", Wdl::BlessedLoss, -112),
            ("6N1/7k/K6p/8/8/8/7N/8 w - - 0 1", Wdl::CursedWin, 102),
            ("8/K7/p3N3/8/8/8/2N5/4k3 b - - 0 1", Wdl::BlessedLoss, -107),
            ("8/8/8/K7/4pP2/8/8/B4k2 b - f3 0 1", Wdl::Draw, 0),
            ("8/2K5/8/1pPk4/8/8/8/3n4 w - b6 0 1", Wdl::Win, 1),
            ("8/8/7K/5pP1/8/4P3/8/6k1 w - f6 0 1", Wdl::Win, 1),
            ("4k3/8/8/1pP5/8/5K2/8/8 w - b6 0 1", Wdl::Draw, 0),
            ("3q4/8/8/1Pp5/8/7K/1k6/8 w - c6 0 1", Wdl::Loss, -2),
            ("4K3/8/8/5P2/Pp6/8/2k5/8 b - a3 0 1", Wdl::Draw, 0),
        ];

        for (fen, wdl, reference_dtz) in cases {
            let reference_dtz: i32 = reference_dtz;
            let mut position = board(fen);
            assert_eq!(
                tablebases.probe_wdl(&mut position),
                Some(wdl),
                "fen {fen:?}, wdl"
            );
            let dtz = tablebases.probe_dtz(&mut position).expect("dtz probed");
            let rounded_down = reference_dtz - reference_dtz.signum();
            assert!(
                dtz == reference_dtz || dtz == rounded_down,
                "fen {fen:?}, dtz {dtz}, reference {reference_dtz}"
            );
            assert_eq!(
                position,
                board(fen),
                "fen {fen:?}, probing restores the board"
            );
        }
    }

    #[test]
    fn refuses_positions_outside_the_tables() {
        let Some(tablebases) = local_tablebases() else {
            return;
        };
        let cases = [
            ("4k2r/8/8/8/8/8/8/4K2R w Kk - 0 1", "castling rights"),
            ("6k1/7p/8/8/8/2K5/3R3r/1R6 w - - 0 1", "six pieces"),
            (crate::board::START_POSITION, "start position"),
        ];

        for (fen, reason) in cases {
            let mut position = board(fen);
            assert_eq!(tablebases.probe_wdl(&mut position), None, "{reason}");
            assert_eq!(tablebases.probe_dtz(&mut position), None, "{reason}");
            assert_eq!(tablebases.root_moves(&mut position), None, "{reason}");
        }
    }

    #[test]
    fn root_moves_keep_the_tablebase_result() {
        let Some(tablebases) = local_tablebases() else {
            return;
        };
        let cases = [
            ("8/8/4k3/8/8/8/1B6/1K2N3 w - - 0 1", Wdl::Win, 2),
            ("8/3k4/8/8/R7/8/1r6/2K5 w - - 0 1", Wdl::Win, 1),
            ("8/8/8/5K2/8/k4q2/6Q1/8 w - - 0 1", Wdl::Win, 1),
            ("3k4/8/8/2B5/4r3/8/2K5/8 w - - 0 1", Wdl::Draw, 0),
            ("8/8/8/3k4/2n5/8/7B/4KB2 w - - 0 1", Wdl::CursedWin, 0),
            ("6k1/8/1R6/K5q1/8/8/8/8 w - - 0 1", Wdl::Loss, 0),
            ("8/8/4k3/8/8/8/1B6/1K2N3 w - - 90 120", Wdl::CursedWin, 0),
        ];

        for (fen, wdl, minimum_excluded) in cases {
            let mut position = board(fen);
            let legal_moves = position.legal_moves().len();
            let root = tablebases
                .root_moves(&mut position)
                .expect("root in tables");
            assert_eq!(root.wdl, wdl, "fen {fen:?}");
            assert!(!root.moves.is_empty(), "fen {fen:?}");
            assert!(
                legal_moves - root.moves.len() >= minimum_excluded,
                "fen {fen:?}, {} of {legal_moves} moves kept",
                root.moves.len()
            );
            for &kept in &root.moves {
                position.make_move(kept);
                let reply = tablebases.probe_wdl(&mut position).expect("reply probed");
                position.unmake_move();
                let expected_reply = match wdl {
                    Wdl::Win | Wdl::CursedWin => reply < Wdl::Draw,
                    Wdl::Draw => reply == Wdl::Draw,
                    Wdl::BlessedLoss | Wdl::Loss => true,
                };
                assert!(expected_reply, "fen {fen:?}, move {kept}, reply {reply:?}");
            }
        }
    }

    #[test]
    fn detects_repetitions_and_the_fifty_move_rule() {
        let start = "4k3/8/8/8/8/2K5/7R/8 w - - 0 1";
        let shuffle = ["c3c4", "e8d8", "c4c3", "d8e8"];
        let cases: [(&str, usize, bool, bool); 5] = [
            (start, 0, false, false),
            (start, 2, false, false),
            (start, 4, true, false),
            (start, 8, true, true),
            ("4k3/8/8/8/8/2K5/7R/8 w - - 100 80", 0, false, true),
        ];

        for (fen, plies, repeated, drawn) in cases {
            let mut position = board(fen);
            for notation in shuffle.iter().cycle().take(plies) {
                let played = find_move(&position, notation);
                assert!(position.make_move(played), "fen {fen:?}, move {notation}");
            }
            assert_eq!(
                repeated_since_zeroing(&position),
                repeated,
                "fen {fen:?}, {plies} plies"
            );
            assert_eq!(
                drawn_by_rule(&mut position),
                drawn,
                "fen {fen:?}, {plies} plies"
            );
        }
    }

    #[test]
    fn opens_only_existing_directories() {
        let cases = [
            ("/nonexistent/arabica/syzygy", false),
            (env!("CARGO_MANIFEST_DIR"), true),
        ];

        for (path, opens) in cases {
            let opened = Tablebases::open(path);
            assert_eq!(opened.is_ok(), opens, "path {path}");
            if let Ok(tablebases) = opened {
                assert!(tablebases.is_empty(), "path {path} holds no tables");
            }
        }
    }
}
