//! The bytes of a profile file. On Linux, a large file on a local file system is mapped, thus the process does not keep a second copy of the page cache.
//!
//! A read lease keeps the mapped bytes valid. Before another process writes or truncates the file, the system sends `SIGIO` and waits. Then a thread copies the bytes
//! into private memory at the same address, and gives back the lease.

use std::fs::File;

pub enum Input {
    Owned(Vec<u8>),
    #[cfg(target_os = "linux")]
    Mapped(Mapping),
}

impl std::ops::Deref for Input {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            #[cfg(target_os = "linux")]
            Self::Mapped(mapping) => unsafe {
                std::slice::from_raw_parts(
                    std::ptr::with_exposed_provenance(mapping.start),
                    mapping.len,
                )
            },
        }
    }
}

impl From<Vec<u8>> for Input {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Owned(bytes)
    }
}

/// Maps `file` if it is large, on a local file system, and has no writer. Else gives back `file`.
#[cfg(target_os = "linux")]
pub fn map(file: File, len: u64) -> Result<Input, File> {
    use std::os::fd::AsRawFd;
    let fd = file.as_raw_fd();
    let mut info: libc::statfs = unsafe { std::mem::zeroed() };
    let local = unsafe { libc::fstatfs(fd, &raw mut info) } == 0
        && [
            libc::TMPFS_MAGIC,
            libc::EXT4_SUPER_MAGIC,
            libc::XFS_SUPER_MAGIC,
            libc::BTRFS_SUPER_MAGIC,
        ]
        .contains(&info.f_type);
    if len < MAP_BYTES || !local || !*LISTENING {
        return Err(file);
    }
    // The lock makes sure that the thread sees the lease break of a new mapping.
    let mut leases = LEASES.lock().unwrap();
    if unsafe { libc::fcntl(fd, libc::F_SETLEASE, libc::F_RDLCK) } != 0
        || !file.metadata().is_ok_and(|meta| meta.len() == len)
    {
        return Err(file);
    }
    let start = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len as usize,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    if start == libc::MAP_FAILED {
        return Err(file);
    }
    let mapping = Mapping {
        start: start.expose_provenance(),
        len: len as usize,
        _file: file,
    };
    leases.push(Lease {
        fd,
        start: mapping.start,
        len: mapping.len,
    });
    Ok(Input::Mapped(mapping))
}

#[cfg(not(target_os = "linux"))]
pub fn map(file: File, _: u64) -> Result<Input, File> {
    Err(file)
}

/// From this size, `map` maps a file.
#[cfg(target_os = "linux")]
const MAP_BYTES: u64 = 16 << 20;

#[cfg(target_os = "linux")]
pub struct Mapping {
    start: usize,
    len: usize,
    /// The lease stays while the file is open.
    _file: File,
}

#[cfg(target_os = "linux")]
impl Drop for Mapping {
    fn drop(&mut self) {
        let mut leases = LEASES.lock().unwrap();
        leases.retain(|lease| lease.start != self.start);
        unsafe { libc::munmap(std::ptr::with_exposed_provenance_mut(self.start), self.len) };
    }
}

/// A mapping that still shows the page cache.
#[cfg(target_os = "linux")]
struct Lease {
    fd: libc::c_int,
    start: usize,
    len: usize,
}

#[cfg(target_os = "linux")]
impl Lease {
    /// Puts a private copy of the bytes at the same address, then gives back the lease. Readers see the same bytes during the change.
    fn copy(&self) -> bool {
        let copy = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                self.len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if copy == libc::MAP_FAILED {
            return false;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::with_exposed_provenance::<u8>(self.start),
                copy.cast::<u8>(),
                self.len,
            );
            libc::mprotect(copy, self.len, libc::PROT_READ);
            libc::mremap(
                copy,
                self.len,
                self.len,
                libc::MREMAP_MAYMOVE | libc::MREMAP_FIXED,
                std::ptr::with_exposed_provenance_mut::<libc::c_void>(self.start),
            );
            libc::fcntl(self.fd, libc::F_SETLEASE, libc::F_UNLCK);
        }
        true
    }
}

#[cfg(target_os = "linux")]
static LEASES: std::sync::Mutex<Vec<Lease>> = std::sync::Mutex::new(Vec::new());

/// The write end of the pipe that wakes the thread of the leases.
#[cfg(target_os = "linux")]
static WAKE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// Whether the thread of the leases runs. Without it, a lease break stops the process.
#[cfg(target_os = "linux")]
static LISTENING: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    let mut pipe = [0; 2];
    if unsafe { libc::pipe2(pipe.as_mut_ptr(), libc::O_CLOEXEC) } != 0
        || unsafe { libc::fcntl(pipe[1], libc::F_SETFL, libc::O_NONBLOCK) } != 0
    {
        return false;
    }
    WAKE.store(pipe[1], std::sync::atomic::Ordering::Relaxed);
    let listen = move || loop {
        let mut byte = 0u8;
        if unsafe { libc::read(pipe[0], (&raw mut byte).cast(), 1) } < 0
            && std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted
        {
            return;
        }
        // During a lease break, `F_GETLEASE` gives `F_UNLCK`. A lease that the thread cannot copy stays, and the writer waits for the lease break time of the system.
        LEASES.lock().unwrap().retain(|lease| unsafe { libc::fcntl(lease.fd, libc::F_GETLEASE) } == libc::F_RDLCK || !lease.copy());
    };
    if std::thread::Builder::new()
        .name("leases".into())
        .spawn(listen)
        .is_err()
    {
        return false;
    }
    let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
    action.sa_sigaction = wake as extern "C" fn(libc::c_int) as libc::sighandler_t;
    action.sa_flags = libc::SA_RESTART;
    unsafe { libc::sigaction(libc::SIGIO, &raw const action, std::ptr::null_mut()) == 0 }
});

#[cfg(target_os = "linux")]
extern "C" fn wake(_: libc::c_int) {
    let byte = 0u8;
    unsafe {
        let errno = *libc::__errno_location();
        libc::write(
            WAKE.load(std::sync::atomic::Ordering::Relaxed),
            (&raw const byte).cast(),
            1,
        );
        *libc::__errno_location() = errno;
    }
}
