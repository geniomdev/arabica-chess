mod attacks;
mod board;
mod types;

use board::{Board, MoveList, START_POSITION};

fn main() {
    attacks::init();

    let board: board::Board = START_POSITION.parse().expect("valid position");
    println!("{START_POSITION}\n");
    println!("{board}\n");

    let mut moves = MoveList::new();
    board.generate_pseudo_legal(&mut moves);

    println!("Moves: {}", moves.as_slice().len());
    for (number, candidate) in moves.as_slice().iter().enumerate() {
        println!("{:>2}. {candidate}", number + 1);
    }
}
