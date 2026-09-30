mod attacks;
mod board;
mod eval;
mod search;
mod tt;
mod types;
mod uci;

fn main() {
    attacks::init();
    uci::run();
}
