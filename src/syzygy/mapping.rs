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

#[cfg(any(unix, windows))]
struct MappedBytes {
    address: *mut std::ffi::c_void,
    len: usize,
}

#[cfg(any(unix, windows))]
unsafe impl Send for MappedBytes {}
#[cfg(any(unix, windows))]
unsafe impl Sync for MappedBytes {}

#[cfg(unix)]
mod system {
    use std::ffi::{c_int, c_long, c_void};
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;

    const PROT_READ: c_int = 1;
    const MAP_SHARED: c_int = 1;
    const MAP_FAILED: *mut c_void = !0 as *mut c_void;

    unsafe extern "C" {
        fn mmap(
            address: *mut c_void,
            len: usize,
            protection: c_int,
            flags: c_int,
            descriptor: c_int,
            offset: c_long,
        ) -> *mut c_void;
        fn munmap(address: *mut c_void, len: usize) -> c_int;
    }

    pub fn map(file: &File, len: usize) -> io::Result<*mut c_void> {
        let address = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ,
                MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if address == MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(address)
    }

    pub unsafe fn unmap(address: *mut c_void, len: usize) {
        unsafe {
            munmap(address, len);
        }
    }
}

#[cfg(windows)]
mod system {
    use std::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::windows::io::AsRawHandle;

    const PAGE_READONLY: u32 = 0x02;
    const FILE_MAP_READ: u32 = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileMappingW(
            file: *mut c_void,
            attributes: *const c_void,
            protection: u32,
            maximum_size_high: u32,
            maximum_size_low: u32,
            name: *const u16,
        ) -> *mut c_void;
        fn MapViewOfFile(
            mapping: *mut c_void,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            len: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(address: *const c_void) -> i32;
        fn CloseHandle(handle: *mut c_void) -> i32;
    }

    pub fn map(file: &File, len: usize) -> io::Result<*mut c_void> {
        let mapping = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        if mapping.is_null() {
            return Err(io::Error::last_os_error());
        }
        let address = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len) };
        let error = address.is_null().then(io::Error::last_os_error);
        unsafe {
            CloseHandle(mapping);
        }
        match error {
            Some(error) => Err(error),
            None => Ok(address),
        }
    }

    pub unsafe fn unmap(address: *mut c_void, _len: usize) {
        unsafe {
            UnmapViewOfFile(address);
        }
    }
}

#[cfg(any(unix, windows))]
impl MappedBytes {
    fn new(file: &File, len: usize) -> io::Result<Self> {
        if len == 0 {
            return Ok(Self {
                address: std::ptr::null_mut(),
                len,
            });
        }
        Ok(Self {
            address: system::map(file, len)?,
            len,
        })
    }

    fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.address.cast::<u8>(), self.len) }
    }
}

#[cfg(any(unix, windows))]
impl Drop for MappedBytes {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe {
                system::unmap(self.address, self.len);
            }
        }
    }
}

#[cfg(not(any(unix, windows)))]
struct MappedBytes(Vec<u8>);

#[cfg(not(any(unix, windows)))]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_file_contents() {
        let large: Vec<u8> = (0..100_000u32).map(|index| (index % 251) as u8).collect();
        let cases: [(&str, &[u8]); 3] = [("empty", &[]), ("small", b"arabica"), ("large", &large)];

        for (name, contents) in cases {
            let path = std::env::temp_dir()
                .join(format!("arabica-mapping-{}-{name}.bin", std::process::id()));
            std::fs::write(&path, contents).expect("temporary file is writable");
            let mapped = Mapping::open(&path).map(|mapping| mapping.bytes().to_vec());
            std::fs::remove_file(&path).expect("temporary file is removable");
            assert_eq!(mapped.expect("file maps"), contents, "case {name}");
        }
    }

    #[test]
    fn missing_files_fail_to_open() {
        let path = std::env::temp_dir().join("arabica-mapping-missing/none.bin");
        assert!(Mapping::open(&path).is_err());
    }
}
