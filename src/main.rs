mod attacks;
mod board;
mod types;

use board::{Board, MoveList, START_POSITION};

fn main() {
    attacks::init();

    let mut board: Board = START_POSITION.parse().expect("valid position");
    println!("{START_POSITION}\n");
    println!("{board}\n");

    let mut moves = MoveList::new();
    board.generate_pseudo_legal(&mut moves);

    let mut number = 0;
    for &candidate in moves.as_slice() {
        if !board.make_move(candidate) {
            continue;
        }
        number += 1;
        println!("{number:>2}. {candidate}   {}\n", board.to_fen());
        println!("{board}\n");
        board.unmake_move();
    }
}
