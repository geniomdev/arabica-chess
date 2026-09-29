mod board;
mod types;

use types::{Bitboard, EveryPiece, EverySide, EverySquare, Piece};

fn main() {
    let board: board::Board = board::KIWIPETE.parse().expect("valid position");
    println!("{}\n", board::KIWIPETE);
    println!("{board}\n");
    println!("{:#?}\n", board.state);
    println!("Bitboard: {}", std::mem::size_of::<types::Bitboard>());
    println!("pieces: {}", std::mem::size_of::<EveryPiece<Bitboard>>());
    println!("colors: {}", std::mem::size_of::<EverySide<Bitboard>>());
    println!(
        "mailbox: {}",
        std::mem::size_of::<EverySquare<Option<Piece>>>()
    );
    println!("GameState: {}", std::mem::size_of::<board::GameState>());
    println!("History: {}", std::mem::size_of::<board::History>());
}
