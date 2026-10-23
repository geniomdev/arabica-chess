use std::io::{self, IsTerminal};
use std::time::{Duration, Instant};

const BAR_WIDTH: usize = 30;
const TERMINAL_INTERVAL: Duration = Duration::from_secs(1);
const LOG_INTERVAL: Duration = Duration::from_secs(60);
const CLEAR_TO_LINE_END: &str = "\x1b[K";

pub fn percent(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

pub fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    format!(
        "{}:{:02}:{:02}",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60
    )
}

pub fn bar(done: u64, total: u64) -> String {
    let filled = (BAR_WIDTH as u64 * done.min(total) / total.max(1)) as usize;
    format!("[{}{}]", "#".repeat(filled), "-".repeat(BAR_WIDTH - filled))
}

pub fn eta(done: u64, total: u64, elapsed: Duration) -> String {
    if done == 0 {
        return String::from("-:--:--");
    }
    let remaining = total.saturating_sub(done) as f64 / done as f64;
    format_duration(elapsed.mul_f64(remaining))
}

pub struct Ticker {
    interactive: bool,
    interval: Duration,
    last: Instant,
}

impl Default for Ticker {
    fn default() -> Self {
        Self::new()
    }
}

impl Ticker {
    pub fn new() -> Self {
        let interactive = io::stderr().is_terminal();
        Self {
            interactive,
            interval: if interactive {
                TERMINAL_INTERVAL
            } else {
                LOG_INTERVAL
            },
            last: Instant::now(),
        }
    }

    pub fn due(&mut self) -> bool {
        let due = self.last.elapsed() >= self.interval;
        if due {
            self.last = Instant::now();
        }
        due
    }

    pub fn show(&self, line: &str) {
        if self.interactive {
            eprint!("\r{line}{CLEAR_TO_LINE_END}");
        } else {
            eprintln!("{line}");
        }
    }

    pub fn finish(&self, line: &str) {
        if self.interactive {
            eprintln!("\r{line}{CLEAR_TO_LINE_END}");
        } else {
            eprintln!("{line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_durations() {
        let cases = [
            (0, "0:00:00"),
            (59, "0:00:59"),
            (61, "0:01:01"),
            (3_600, "1:00:00"),
            (45_296, "12:34:56"),
            (100 * 3_600, "100:00:00"),
        ];

        for (seconds, expected) in cases {
            assert_eq!(
                format_duration(Duration::from_secs(seconds)),
                expected,
                "{seconds} s"
            );
        }
    }

    #[test]
    fn renders_bars_and_estimates() {
        let cases = [
            (0, 100, 0, "[------------------------------]", "-:--:--"),
            (25, 100, 100, "[#######-----------------------]", "0:05:00"),
            (100, 100, 50, "[##############################]", "0:00:00"),
            (150, 100, 50, "[##############################]", "0:00:00"),
        ];

        for (done, total, seconds, expected_bar, expected_eta) in cases {
            let elapsed = Duration::from_secs(seconds);
            assert_eq!(bar(done, total), expected_bar, "{done}/{total}");
            assert_eq!(eta(done, total, elapsed), expected_eta, "{done}/{total}");
        }
    }
}
