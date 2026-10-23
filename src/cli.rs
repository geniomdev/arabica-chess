use std::io;

pub fn parse_argument<T: std::str::FromStr>(
    value: Option<&String>,
    name: &str,
    default: T,
) -> io::Result<T> {
    value.map_or(Ok(default), |text| {
        text.parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid {name}: {text}"),
            )
        })
    })
}

pub fn usage_error(usage: &str) -> io::Error {
    eprintln!("usage: {usage}");
    io::Error::new(io::ErrorKind::InvalidInput, "missing arguments")
}

pub fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}
