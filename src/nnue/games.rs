use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;

use super::data::{Outcome, Record};
use crate::board::{Board, MoveList};
use crate::search::MATE_BOUND;
use crate::types::Move;

pub const MAGIC: &[u8; 8] = b"ARBGAME1";
pub const EXTENSION: &str = "games";

const PLY_BYTES: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ply {
    pub best_move: Move,
    pub score: i16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Filter {
    pub skipped_plies: usize,
    pub max_score: i32,
}

impl Filter {
    pub const DEFAULT: Self = Self {
        skipped_plies: 6,
        max_score: MATE_BOUND - 1,
    };

    pub fn accepts(&self, ply_index: usize, board: &Board, ply: Ply) -> bool {
        ply_index >= self.skipped_plies
            && i32::from(ply.score).abs() <= self.max_score
            && !ply.best_move.is_capture()
            && !ply.best_move.is_promotion()
            && !board.in_check()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Game {
    pub start: String,
    pub white_outcome: Outcome,
    pub plies: Vec<Ply>,
}

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn read_bytes<const N: usize>(reader: &mut impl Read) -> io::Result<[u8; N]> {
    let mut bytes = [0; N];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn truncated(error: io::Error) -> io::Error {
    if error.kind() == io::ErrorKind::UnexpectedEof {
        invalid_data("the last game is truncated".to_owned())
    } else {
        error
    }
}

impl Game {
    pub fn write(&self, writer: &mut impl Write) -> io::Result<()> {
        let start_len = u8::try_from(self.start.len())
            .map_err(|_| invalid_data(format!("start position {:?} is too long", self.start)))?;
        let plies = u16::try_from(self.plies.len())
            .map_err(|_| invalid_data(format!("{} plies do not fit a game", self.plies.len())))?;
        let mut bytes = Vec::with_capacity(4 + self.start.len() + self.plies.len() * PLY_BYTES);
        bytes.push(self.white_outcome as u8);
        bytes.push(start_len);
        bytes.extend_from_slice(self.start.as_bytes());
        bytes.extend_from_slice(&plies.to_le_bytes());
        for ply in &self.plies {
            bytes.extend_from_slice(&ply.best_move.raw().to_le_bytes());
            bytes.extend_from_slice(&ply.score.to_le_bytes());
        }
        writer.write_all(&bytes)
    }

    pub fn read(reader: &mut impl Read) -> io::Result<Option<Self>> {
        let mut outcome = [0];
        if reader.read(&mut outcome)? == 0 {
            return Ok(None);
        }
        let white_outcome = Outcome::from_byte(outcome[0])
            .ok_or_else(|| invalid_data(format!("game outcome byte {}", outcome[0])))?;
        let [start_len] = read_bytes::<1>(reader).map_err(truncated)?;
        let mut start = vec![0; usize::from(start_len)];
        reader.read_exact(&mut start).map_err(truncated)?;
        let start = String::from_utf8(start)
            .map_err(|_| invalid_data("start position is not UTF-8".to_owned()))?;
        let plies = u16::from_le_bytes(read_bytes(reader).map_err(truncated)?);
        let mut bytes = vec![0; usize::from(plies) * PLY_BYTES];
        reader.read_exact(&mut bytes).map_err(truncated)?;
        let plies = bytes
            .as_chunks::<PLY_BYTES>()
            .0
            .iter()
            .map(|chunk| Ply {
                best_move: Move::from_raw(u16::from_le_bytes([chunk[0], chunk[1]])),
                score: i16::from_le_bytes([chunk[2], chunk[3]]),
            })
            .collect();
        Ok(Some(Self {
            start,
            white_outcome,
            plies,
        }))
    }

    pub fn records(&self, filter: &Filter) -> io::Result<Vec<Record>> {
        let mut board: Board = self
            .start
            .parse()
            .map_err(|error| invalid_data(format!("start position {:?}: {error}", self.start)))?;
        let mut records = Vec::new();
        for (index, &ply) in self.plies.iter().enumerate() {
            if filter.accepts(index, &board, ply) {
                let outcome = self.white_outcome.for_side(board.state.active_color);
                records.push(Record::new(&board, ply.score, outcome));
            }
            let mut moves = MoveList::new();
            board.generate_pseudo_legal(&mut moves);
            let legal = moves.as_slice().contains(&ply.best_move) && board.make_move(ply.best_move);
            if !legal {
                return Err(invalid_data(format!(
                    "illegal move {} at ply {index} after {:?}",
                    ply.best_move, self.start
                )));
            }
        }
        Ok(records)
    }
}

pub fn write_header(writer: &mut impl Write) -> io::Result<()> {
    writer.write_all(MAGIC)
}

pub fn read_header(reader: &mut impl Read) -> io::Result<bool> {
    let mut magic = [0; MAGIC.len()];
    let mut filled = 0;
    while filled < magic.len() {
        match reader.read(&mut magic[filled..])? {
            0 => return Ok(false),
            read => filled += read,
        }
    }
    Ok(&magic == MAGIC)
}

pub fn is_games_file(path: &Path) -> io::Result<bool> {
    read_header(&mut File::open(path)?)
}

pub struct GameReader<R> {
    reader: R,
}

impl GameReader<BufReader<File>> {
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        if !read_header(&mut reader)? {
            return Err(invalid_data(format!(
                "{} is not an arabica games file",
                path.display()
            )));
        }
        Ok(Self { reader })
    }
}

impl<R: Read> Iterator for GameReader<R> {
    type Item = io::Result<Game>;

    fn next(&mut self) -> Option<Self::Item> {
        Game::read(&mut self.reader).transpose()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Contents {
    pub games: u64,
    pub plies: u64,
}

pub fn contents(path: &Path) -> io::Result<Contents> {
    let mut contents = Contents::default();
    for game in GameReader::open(path)? {
        contents.games += 1;
        contents.plies += game?.plies.len() as u64;
    }
    Ok(contents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::testing::find_move;
    use crate::board::{KIWIPETE, START_POSITION};

    fn game(start: &str, white_outcome: Outcome, line: &[(&str, i16)]) -> Game {
        let mut board: Board = start.parse().expect("valid start");
        let plies = line
            .iter()
            .map(|&(notation, score)| {
                let best_move = find_move(&board, notation);
                assert!(board.make_move(best_move), "{notation} is legal");
                Ply { best_move, score }
            })
            .collect();
        Game {
            start: start.to_owned(),
            white_outcome,
            plies,
        }
    }

    fn games() -> Vec<Game> {
        vec![
            game(
                START_POSITION,
                Outcome::Win,
                &[("e2e4", 30), ("e7e5", -25), ("g1f3", 35), ("b8c6", -30)],
            ),
            game(KIWIPETE, Outcome::Draw, &[("e2a6", -20), ("b4c3", 15)]),
            game("4k3/8/8/8/8/8/8/4K2R w K - 0 1", Outcome::Loss, &[]),
            game(
                "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
                Outcome::Win,
                &[("b7b8q", 900), ("e8d7", -30_900)],
            ),
        ]
    }

    #[test]
    fn games_round_trip_through_bytes() {
        let games = games();
        let mut bytes = Vec::new();
        write_header(&mut bytes).expect("header written");
        for game in &games {
            game.write(&mut bytes).expect("game written");
        }
        let mut reader = bytes.as_slice();
        assert!(read_header(&mut reader).expect("header read"));
        let read: Vec<Game> = GameReader { reader }
            .collect::<io::Result<_>>()
            .expect("games read");
        assert_eq!(read, games);
    }

    #[test]
    fn rejects_foreign_and_truncated_data() {
        let mut game_bytes = MAGIC.to_vec();
        games()[0].write(&mut game_bytes).expect("game written");
        let cases: [(&str, Vec<u8>, bool, Option<usize>); 7] = [
            ("empty", Vec::new(), false, None),
            ("short magic", b"ARBGAME".to_vec(), false, None),
            ("checkpoint", b"ARBNNUE1".to_vec(), false, None),
            ("header only", MAGIC.to_vec(), true, Some(0)),
            ("one game", game_bytes.clone(), true, Some(1)),
            (
                "truncated game",
                game_bytes[..game_bytes.len() - 1].to_vec(),
                true,
                None,
            ),
            (
                "bad outcome",
                [MAGIC.as_slice(), &[3, 0, 0, 0]].concat(),
                true,
                None,
            ),
        ];

        for (name, bytes, header, games) in cases {
            let mut reader = bytes.as_slice();
            assert_eq!(
                read_header(&mut reader).expect("header read"),
                header,
                "{name}"
            );
            if header {
                let read: io::Result<Vec<Game>> = GameReader { reader }.collect();
                assert_eq!(read.ok().map(|games| games.len()), games, "{name}");
            }
        }
    }

    #[test]
    fn replays_games_into_filtered_records() {
        let kept = |filter: Filter, game: &Game| -> Vec<(i16, Outcome)> {
            game.records(&filter)
                .expect("legal game")
                .iter()
                .map(|record| (record.score, record.outcome))
                .collect()
        };
        let every = Filter {
            skipped_plies: 0,
            ..Filter::DEFAULT
        };
        let capped = Filter {
            max_score: 30,
            ..every
        };
        let opening = &games()[0];
        let captures = &games()[1];
        let promotion = &games()[3];
        let check = game(
            "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
            Outcome::Win,
            &[("a1a8", 500), ("e8e7", -520), ("a8a7", 510)],
        );
        let cases = [
            (Filter::DEFAULT, opening, vec![]),
            (
                every,
                opening,
                vec![
                    (30, Outcome::Win),
                    (-25, Outcome::Loss),
                    (35, Outcome::Win),
                    (-30, Outcome::Loss),
                ],
            ),
            (
                capped,
                opening,
                vec![
                    (30, Outcome::Win),
                    (-25, Outcome::Loss),
                    (-30, Outcome::Loss),
                ],
            ),
            (every, captures, vec![]),
            (every, promotion, vec![]),
            (
                every,
                &check,
                vec![(500, Outcome::Win), (510, Outcome::Win)],
            ),
        ];

        for (filter, game, expected) in cases {
            assert_eq!(
                kept(filter, game),
                expected,
                "{filter:?}, start {:?}",
                game.start
            );
        }
    }

    #[test]
    fn rejects_games_with_illegal_moves() {
        let mut broken = games()[0].clone();
        broken.plies.swap(0, 1);
        let mut bad_start = games()[0].clone();
        bad_start.start = "not a fen".to_owned();
        let cases = [
            (games()[0].clone(), true),
            (broken, false),
            (bad_start, false),
        ];

        for (game, valid) in cases {
            assert_eq!(game.records(&Filter::DEFAULT).is_ok(), valid, "{game:?}");
        }
    }
}
