use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::thread;

use arabica::nnue::data::{RECORD_BYTES, Record};
use arabica::nnue::games::{Filter, Game, GameReader, is_games_file};

use super::shard::RawRecord;

const GAMES_PER_BLOCK: usize = 1 << 16;

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub struct Dataset {
    pub records: Vec<RawRecord>,
    pub games: Option<GameCounts>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GameCounts {
    pub games: u64,
    pub plies: u64,
}

pub fn load(path: &Path, filter: &Filter, threads: usize) -> io::Result<Dataset> {
    if is_games_file(path)? {
        let (records, counts) = load_games(path, filter, threads)?;
        Ok(Dataset {
            records,
            games: Some(counts),
        })
    } else {
        Ok(Dataset {
            records: load_positions(path)?,
            games: None,
        })
    }
}

fn load_positions(path: &Path) -> io::Result<Vec<RawRecord>> {
    let mut file = File::open(path)?;
    let bytes = file.metadata()?.len() as usize;
    if !bytes.is_multiple_of(RECORD_BYTES) {
        return Err(invalid_data(format!(
            "{} is {bytes} bytes, not a whole number of {RECORD_BYTES}-byte records",
            path.display()
        )));
    }
    let mut records = vec![[0; RECORD_BYTES]; bytes / RECORD_BYTES];
    file.read_exact(records.as_flattened_mut())?;
    if let Some(index) = records
        .iter()
        .position(|bytes| Record::from_bytes(bytes).is_none())
    {
        return Err(invalid_data(format!(
            "{}: record {index} is malformed",
            path.display()
        )));
    }
    Ok(records)
}

fn replay(games: &[Game], filter: &Filter) -> io::Result<Vec<RawRecord>> {
    let mut records = Vec::new();
    for game in games {
        records.extend(game.records(filter)?.iter().map(|record| record.to_bytes()));
    }
    Ok(records)
}

fn replay_blocks(
    path: &Path,
    filter: &Filter,
    threads: usize,
    mut consume: impl FnMut(Vec<RawRecord>),
) -> io::Result<GameCounts> {
    let mut reader = GameReader::open(path)?;
    let mut counts = GameCounts::default();
    loop {
        let block: Vec<Game> = reader
            .by_ref()
            .take(GAMES_PER_BLOCK)
            .collect::<io::Result<_>>()?;
        if block.is_empty() {
            return Ok(counts);
        }
        counts.games += block.len() as u64;
        counts.plies += block
            .iter()
            .map(|game| game.plies.len() as u64)
            .sum::<u64>();
        let parts = thread::scope(|scope| {
            let workers: Vec<_> = block
                .chunks(block.len().div_ceil(threads.max(1)))
                .map(|games| scope.spawn(move || replay(games, filter)))
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().expect("replay worker finished"))
                .collect::<io::Result<Vec<_>>>()
        })?;
        for part in parts {
            consume(part);
        }
    }
}

fn load_games(
    path: &Path,
    filter: &Filter,
    threads: usize,
) -> io::Result<(Vec<RawRecord>, GameCounts)> {
    let mut kept = 0;
    replay_blocks(path, filter, threads, |part| kept += part.len())?;
    let mut records = Vec::with_capacity(kept);
    let counts = replay_blocks(path, filter, threads, |part| records.extend(part))?;
    if records.len() != kept {
        return Err(invalid_data(format!(
            "{} changed while loading: {kept} then {} positions",
            path.display(),
            records.len()
        )));
    }
    Ok((records, counts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::board;
    use arabica::board::START_POSITION;
    use arabica::nnue::data::Outcome;
    use arabica::nnue::games::{Ply, write_header};

    fn scratch_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("arabica-{}-{name}", std::process::id()))
    }

    fn opening_game(white_outcome: Outcome) -> Game {
        let mut position = board(START_POSITION);
        let plies = [
            "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5",
        ]
        .iter()
        .enumerate()
        .map(|(index, notation)| {
            let best_move = position.parse_move(notation).expect("generated move");
            assert!(position.make_move(best_move), "{notation} is legal");
            Ply {
                best_move,
                score: 10 * index as i16,
            }
        })
        .collect();
        Game {
            start: START_POSITION.to_owned(),
            white_outcome,
            plies,
        }
    }

    fn games_file(games: &[Game]) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_header(&mut bytes).expect("header written");
        for game in games {
            game.write(&mut bytes).expect("game written");
        }
        bytes
    }

    #[test]
    fn loads_position_and_game_files() {
        let valid = Record::new(&board(START_POSITION), 0, Outcome::Draw).to_bytes();
        let mut broken = valid;
        broken[26] = 9;
        let games = [opening_game(Outcome::Win), opening_game(Outcome::Loss)];
        let mut illegal = opening_game(Outcome::Draw);
        illegal.plies.swap(0, 1);
        let full = games_file(&games);
        type Loaded = Option<(usize, Option<GameCounts>)>;
        let cases: [(&str, Vec<u8>, Loaded); 8] = [
            ("empty", Vec::new(), Some((0, None))),
            ("two", [valid, valid].concat(), Some((2, None))),
            ("partial", valid[..31].to_vec(), None),
            ("broken", [valid, broken].concat(), None),
            (
                "games",
                full.clone(),
                Some((
                    4,
                    Some(GameCounts {
                        games: 2,
                        plies: 16,
                    }),
                )),
            ),
            (
                "header only",
                games_file(&[]),
                Some((0, Some(GameCounts::default()))),
            ),
            ("truncated games", full[..full.len() - 1].to_vec(), None),
            ("illegal game", games_file(&[illegal]), None),
        ];

        for (name, bytes, expected) in cases {
            let path = scratch_path(&format!("{name}.data"));
            std::fs::write(&path, bytes).expect("file written");
            let loaded = load(&path, &Filter::DEFAULT, 3)
                .ok()
                .map(|dataset| (dataset.records.len(), dataset.games));
            std::fs::remove_file(&path).expect("file removed");
            assert_eq!(loaded, expected, "{name}");
        }
    }

    #[test]
    fn game_records_follow_the_filter_and_outcomes() {
        let games = [opening_game(Outcome::Win), opening_game(Outcome::Draw)];
        let path = scratch_path("filtered.games");
        std::fs::write(&path, games_file(&games)).expect("file written");
        let every = Filter {
            skipped_plies: 0,
            ..Filter::DEFAULT
        };
        let cases = [
            (
                Filter::DEFAULT,
                1,
                vec![
                    (60, Outcome::Win),
                    (70, Outcome::Loss),
                    (60, Outcome::Draw),
                    (70, Outcome::Draw),
                ],
            ),
            (
                Filter {
                    max_score: 20,
                    ..every
                },
                12,
                vec![
                    (0, Outcome::Win),
                    (10, Outcome::Loss),
                    (20, Outcome::Win),
                    (0, Outcome::Draw),
                    (10, Outcome::Draw),
                    (20, Outcome::Draw),
                ],
            ),
        ];

        for (filter, threads, expected) in cases {
            let dataset = load(&path, &filter, threads).expect("games loaded");
            let kept: Vec<(i16, Outcome)> = dataset
                .records
                .iter()
                .map(|bytes| {
                    let record = Record::from_bytes(bytes).expect("valid record");
                    (record.score, record.outcome)
                })
                .collect();
            assert_eq!(kept, expected, "{filter:?}, {threads} threads");
        }
        std::fs::remove_file(&path).expect("file removed");
    }
}
