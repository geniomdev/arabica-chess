use std::fs::File;
use std::io;
use std::path::Path;

pub struct Mapping {
    bytes: MappedBytes,
}

impl Mapping {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "file too large to map"))?;
        Ok(Self {
            bytes: MappedBytes::new(&file, len)?,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }
}

#[cfg(unix)]
struct MappedBytes {
    address: *mut std::ffi::c_void,
    len: usize,
}

#[cfg(unix)]
unsafe impl Send for MappedBytes {}
#[cfg(unix)]
unsafe impl Sync for MappedBytes {}

#[cfg(unix)]
mod system {
    use std::ffi::{c_int, c_long, c_void};

    pub const PROT_READ: c_int = 1;
    pub const MAP_SHARED: c_int = 1;
    pub const MAP_FAILED: *mut c_void = !0 as *mut c_void;

    unsafe extern "C" {
        pub fn mmap(
            address: *mut c_void,
            len: usize,
            protection: c_int,
            flags: c_int,
            descriptor: c_int,
            offset: c_long,
        ) -> *mut c_void;
        pub fn munmap(address: *mut c_void, len: usize) -> c_int;
    }
}

#[cfg(unix)]
impl MappedBytes {
    fn new(file: &File, len: usize) -> io::Result<Self> {
        use std::os::fd::AsRawFd;

        if len == 0 {
            return Ok(Self {
                address: std::ptr::null_mut(),
                len,
            });
        }
        let address = unsafe {
            system::mmap(
                std::ptr::null_mut(),
                len,
                system::PROT_READ,
                system::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if address == system::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { address, len })
    }

    fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.address.cast::<u8>(), self.len) }
    }
}

#[cfg(unix)]
impl Drop for MappedBytes {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe {
                system::munmap(self.address, self.len);
            }
        }
    }
}

#[cfg(not(unix))]
struct MappedBytes(Vec<u8>);

#[cfg(not(unix))]
impl MappedBytes {
    fn new(file: &File, len: usize) -> io::Result<Self> {
        use std::io::Read;

        let mut bytes = Vec::with_capacity(len);
        let mut reader = file;
        reader.read_to_end(&mut bytes)?;
        Ok(Self(bytes))
    }

    fn as_slice(&self) -> &[u8] {
        &self.0
    }
}
