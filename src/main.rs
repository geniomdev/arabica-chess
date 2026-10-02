mod attacks;
mod board;
mod eval;
mod search;
mod strength;
mod tt;
mod tune;
mod types;
mod uci;

fn main() -> std::io::Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.split_first() {
        Some((command, rest)) if command == "tune" => tune::run(rest),
        _ => {
            uci::run();
            Ok(())
        }
    }
}
