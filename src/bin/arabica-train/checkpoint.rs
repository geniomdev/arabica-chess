use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use super::model::{Adam, Model, PARAMETERS};

const MAGIC: &[u8; 8] = b"ARBNNUE1";
const FLOAT_BYTES: usize = std::mem::size_of::<f32>();

#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    pub epoch: usize,
    pub model: Model,
    pub adam: Adam,
}

fn write_floats(writer: &mut impl Write, values: &[f32]) -> io::Result<()> {
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_floats(reader: &mut impl Read) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0; PARAMETERS * FLOAT_BYTES];
    reader.read_exact(&mut bytes)?;
    Ok(bytes
        .as_chunks::<FLOAT_BYTES>()
        .0
        .iter()
        .map(|&chunk| f32::from_le_bytes(chunk))
        .collect())
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

pub fn replace_file(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>,
) -> io::Result<()> {
    if let Some(directory) = path.parent() {
        fs::create_dir_all(directory)?;
    }
    let temporary = temporary_path(path);
    let mut writer = BufWriter::new(File::create(&temporary)?);
    write(&mut writer)?;
    writer.into_inner()?.sync_all()?;
    fs::rename(temporary, path)
}

impl Checkpoint {
    pub fn save(&self, path: &Path) -> io::Result<()> {
        replace_file(path, |writer| {
            writer.write_all(MAGIC)?;
            writer.write_all(&(self.epoch as u64).to_le_bytes())?;
            writer.write_all(&self.adam.step.to_le_bytes())?;
            write_floats(writer, &self.model.parameters)?;
            write_floats(writer, &self.adam.first_moments)?;
            write_floats(writer, &self.adam.second_moments)
        })
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut magic = [0; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} is not an arabica training checkpoint", path.display()),
            ));
        }
        let epoch = read_u64(&mut reader)? as usize;
        let step = read_u64(&mut reader)?;
        let parameters = read_floats(&mut reader)?;
        let first_moments = read_floats(&mut reader)?;
        let second_moments = read_floats(&mut reader)?;
        if reader.read(&mut [0])? != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} has trailing bytes", path.display()),
            ));
        }
        Ok(Self {
            epoch,
            model: Model { parameters },
            adam: Adam {
                step,
                first_moments,
                second_moments,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arabica::random::Random;

    fn scratch_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("arabica-{}-{name}", std::process::id()))
    }

    #[test]
    fn checkpoints_round_trip() {
        let mut random = Random::new(3);
        let mut adam = Adam::new();
        adam.step = 12_345;
        for (first, second) in adam.first_moments.iter_mut().zip(&mut adam.second_moments) {
            *first = random.unit() - 0.5;
            *second = random.unit();
        }
        let cases = [("fresh.ckpt", 0, Adam::new()), ("trained.ckpt", 17, adam)];

        for (name, epoch, adam) in cases {
            let path = scratch_path(name);
            let checkpoint = Checkpoint {
                epoch,
                model: Model::random(&mut random),
                adam,
            };
            checkpoint.save(&path).expect("checkpoint saved");
            let loaded = Checkpoint::load(&path).expect("checkpoint loaded");
            fs::remove_file(&path).expect("checkpoint removed");
            assert_eq!(loaded, checkpoint, "{name}");
        }
    }

    #[test]
    fn rejects_foreign_files() {
        let path = scratch_path("foreign.ckpt");
        let cases: [&[u8]; 3] = [b"", b"NOTACKPTxxxxxxxxxxxxxxxx", b"ARBNNUE1\0\0"];

        for bytes in cases {
            fs::write(&path, bytes).expect("file written");
            assert!(Checkpoint::load(&path).is_err(), "bytes {bytes:?}");
        }
        fs::remove_file(&path).expect("file removed");
    }
}
