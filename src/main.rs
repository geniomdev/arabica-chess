mod attacks;
mod bench;
mod board;
mod cli;
mod datagen;
mod eval;
mod history;
mod movepick;
mod nnue;
mod progress;
mod random;
mod search;
mod strength;
mod train;
mod tt;
mod tune;
mod types;
mod uci;

fn main() -> std::io::Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.split_first() {
        Some((command, rest)) if command == "tune" => tune::run(rest),
        Some((command, rest)) if command == "datagen" => datagen::run(rest),
        Some((command, rest)) if command == "train" => train::run(rest),
        Some((command, rest)) if command == "export" => train::export_command(rest),
        _ => {
            uci::run();
            Ok(())
        }
    }
}
