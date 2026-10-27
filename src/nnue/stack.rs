use super::{Accumulator, HIDDEN, Network, active_features, feature_index};
use crate::types::{Bitboard, Color, Piece, SIDES, Square};

const MAX_CHANGES: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Computed,
    Pending,
    Stale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Change {
    side: Color,
    piece: Piece,
    square: Square,
}

#[derive(Clone, Copy)]
struct Changes {
    items: [Change; MAX_CHANGES],
    len: usize,
}

impl Changes {
    const EMPTY: Self = Self {
        items: [Change {
            side: Color::White,
            piece: Piece::Pawn,
            square: Square::A1,
        }; MAX_CHANGES],
        len: 0,
    };

    fn push(&mut self, change: Change) -> bool {
        if self.len == MAX_CHANGES {
            return false;
        }
        self.items[self.len] = change;
        self.len += 1;
        true
    }

    fn rows<'a>(
        &self,
        network: &'a Network,
        perspective: Color,
    ) -> ([&'a [i16; HIDDEN]; MAX_CHANGES], usize) {
        let mut rows = [&network.feature_bias; MAX_CHANGES];
        for (row, change) in rows.iter_mut().zip(&self.items[..self.len]) {
            *row = &network.feature_weights
                [feature_index(perspective, change.side, change.piece, change.square)];
        }
        (rows, self.len)
    }
}

#[derive(Clone, Copy)]
struct Entry {
    accumulators: [Accumulator; SIDES],
    added: Changes,
    removed: Changes,
    status: Status,
}

impl Entry {
    const STALE: Self = Self {
        accumulators: [Accumulator([0; HIDDEN]); SIDES],
        added: Changes::EMPTY,
        removed: Changes::EMPTY,
        status: Status::Stale,
    };
}

#[derive(Clone)]
pub struct AccumulatorStack {
    entries: Vec<Entry>,
    top: usize,
}

impl std::fmt::Debug for AccumulatorStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccumulatorStack")
            .field("top", &self.top)
            .field("status", &self.entries[self.top].status)
            .finish()
    }
}

fn apply(
    parent: &[i16; HIDDEN],
    added: &[&[i16; HIDDEN]],
    removed: &[&[i16; HIDDEN]],
    output: &mut [i16; HIDDEN],
) {
    match (added, removed) {
        ([add], [remove]) => {
            for index in 0..HIDDEN {
                output[index] = parent[index] + add[index] - remove[index];
            }
        }
        ([add], [first, second]) => {
            for index in 0..HIDDEN {
                output[index] = parent[index] + add[index] - first[index] - second[index];
            }
        }
        ([first_add, second_add], [first, second]) => {
            for index in 0..HIDDEN {
                output[index] = parent[index] + first_add[index] + second_add[index]
                    - first[index]
                    - second[index];
            }
        }
        _ => {
            *output = *parent;
            for row in added {
                for index in 0..HIDDEN {
                    output[index] += row[index];
                }
            }
            for row in removed {
                for index in 0..HIDDEN {
                    output[index] -= row[index];
                }
            }
        }
    }
}

impl AccumulatorStack {
    pub fn new() -> Self {
        Self {
            entries: vec![Entry::STALE],
            top: 0,
        }
    }

    pub fn push(&mut self) {
        self.top += 1;
        if self.top == self.entries.len() {
            self.entries.push(Entry::STALE);
        }
        let entry = &mut self.entries[self.top];
        entry.added.len = 0;
        entry.removed.len = 0;
        entry.status = Status::Pending;
    }

    pub fn invalidate(&mut self) {
        for entry in &mut self.entries {
            entry.status = Status::Stale;
        }
    }

    pub fn pop(&mut self) {
        debug_assert!(self.top > 0, "accumulator stack underflow");
        self.top -= 1;
    }

    pub fn add(&mut self, side: Color, piece: Piece, square: Square) {
        let entry = &mut self.entries[self.top];
        if entry.status == Status::Pending
            && !entry.added.push(Change {
                side,
                piece,
                square,
            })
        {
            entry.status = Status::Stale;
        }
    }

    pub fn remove(&mut self, side: Color, piece: Piece, square: Square) {
        let entry = &mut self.entries[self.top];
        if entry.status == Status::Pending
            && !entry.removed.push(Change {
                side,
                piece,
                square,
            })
        {
            entry.status = Status::Stale;
        }
    }

    pub fn shift(&mut self, side: Color, piece: Piece, from: Square, to: Square) {
        self.remove(side, piece, from);
        self.add(side, piece, to);
    }

    pub fn current(
        &mut self,
        network: &Network,
        pieces_of: impl Fn(Color, Piece) -> Bitboard,
    ) -> &[Accumulator; SIDES] {
        let top = self.top;
        let mut base = top;
        while self.entries[base].status == Status::Pending {
            base -= 1;
        }
        if self.entries[base].status == Status::Stale {
            let entry = &mut self.entries[top];
            for perspective in Color::ALL {
                entry.accumulators[perspective.index()] =
                    network.refresh(active_features(perspective, &pieces_of));
            }
            entry.status = Status::Computed;
            return &self.entries[top].accumulators;
        }
        for index in base + 1..=top {
            let (done, rest) = self.entries.split_at_mut(index);
            let parent = &done[index - 1];
            let entry = &mut rest[0];
            for perspective in Color::ALL {
                let (added, added_len) = entry.added.rows(network, perspective);
                let (removed, removed_len) = entry.removed.rows(network, perspective);
                apply(
                    &parent.accumulators[perspective.index()].0,
                    &added[..added_len],
                    &removed[..removed_len],
                    &mut entry.accumulators[perspective.index()].0,
                );
            }
            entry.status = Status::Computed;
        }
        &self.entries[top].accumulators
    }
}
