use arabica::{datagen, uci};

fn main() -> std::io::Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.split_first() {
        Some((command, rest)) if command == "datagen" => datagen::run(rest),
        _ => {
            uci::run();
            Ok(())
        }
    }
}
