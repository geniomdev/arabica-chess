use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::board::{Board, see_value};
use crate::history::{Continuations, History, PieceTo, history_bonus};
use crate::movepick::{MovePicker, captured_piece, moved_piece};
use crate::nnue;
use crate::strength::EvalNoise;
use crate::syzygy::{Tablebases, Wdl};
use crate::tt::{Bound, Hit, Store, TranspositionTable};
use crate::types::{Color, Move, Piece, SQUARES};

pub const INFINITY: i32 = 32_000;
pub const MATE: i32 = 31_000;
pub const MAX_PLY: usize = 128;
pub const MAX_DEPTH: u8 = 64;

pub const MATE_BOUND: i32 = MATE - MAX_PLY as i32;
pub const TABLEBASE_WIN: i32 = MATE_BOUND - 1 - MAX_PLY as i32;
const TABLEBASE_DEPTH_BONUS: i32 = 6;
const TIME_CHECK_INTERVAL: u64 = 2048;
const QUIESCENCE_DEPTH: u8 = 0;
const QUIESCENCE_FUTILITY_MARGIN: i32 = 200;

const TRACKED_QUIETS: usize = 64;
const TRACKED_NOISY: usize = 32;

const REVERSE_FUTILITY_DEPTH: i32 = 8;
const REVERSE_FUTILITY_MARGIN: i32 = 80;
const REVERSE_FUTILITY_IMPROVING_MARGIN: i32 = 40;
const NULL_MOVE_DEPTH: i32 = 3;
const LATE_MOVE_PRUNING_DEPTH: i32 = 8;
const FUTILITY_DEPTH: i32 = 6;
const FUTILITY_BASE: i32 = 100;
const FUTILITY_MARGIN: i32 = 100;
const NOISY_SEE_DEPTH: i32 = 6;
const NOISY_SEE_MARGIN: i32 = 100;
const REDUCTION_DEPTH: i32 = 3;
const REDUCTION_HISTORY_DIVISOR: i32 = 16_384;
const REDUCTION_TABLE_SIZE: usize = 64;
const INTERNAL_REDUCTION_DEPTH: i32 = 4;
const SINGULAR_DEPTH: i32 = 8;
const SINGULAR_TABLE_DEPTH_MARGIN: i32 = 3;
const SINGULAR_BETA_MARGIN: i32 = 2;
const PROBCUT_DEPTH: i32 = 5;
const PROBCUT_MARGIN: i32 = 200;
const PROBCUT_REDUCTION: i32 = 4;
const HISTORY_PRUNING_DEPTH: i32 = 4;
const HISTORY_PRUNING_MARGIN: i32 = 2_048;
const QUIET_SEE_DEPTH: i32 = 8;
const QUIET_SEE_MARGIN: i32 = 60;
const PAWN_KEY_MULTIPLIER: u64 = 0x9E37_79B9_7F4A_7C15;
const BLACK_PAWNS_ROTATION: u32 = 29;

const ASPIRATION_DEPTH: i32 = 4;
const ASPIRATION_WINDOW: i32 = 25;
const ASPIRATION_FULL_WINDOW_THRESHOLD: i32 = 400;

const STABILITY_TIME_SCALES: [f64; 5] = [2.0, 1.4, 1.1, 0.9, 0.8];
const SCORE_DROP_RANGE: (i32, i32) = (-50, 100);
const SCORE_DROP_TIME_DIVISOR: f64 = 200.0;
const BEST_MOVE_NODES_PIVOT: f64 = 1.5;
const BEST_MOVE_NODES_SCALE: f64 = 1.35;
const SOFT_TIME_SCALE_RANGE: (f64, f64) = (0.5, 2.5);

type RootMoveNodes = [[u64; SQUARES]; SQUARES];

static REDUCTIONS: LazyLock<[[i32; REDUCTION_TABLE_SIZE]; REDUCTION_TABLE_SIZE]> =
    LazyLock::new(|| {
        let mut table = [[0; REDUCTION_TABLE_SIZE]; REDUCTION_TABLE_SIZE];
        for (depth, row) in table.iter_mut().enumerate().skip(1) {
            for (move_number, reduction) in row.iter_mut().enumerate().skip(1) {
                let scaled = (depth as f64).ln() * (move_number as f64).ln() / 2.25;
                *reduction = (0.75 + scaled) as i32;
            }
        }
        table
    });

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchLimits {
    pub depth: Option<u8>,
    pub time: Option<Duration>,
    pub soft_time: Option<Duration>,
    pub nodes: Option<u64>,
    pub soft_nodes: Option<u64>,
}

pub struct Iteration<'a> {
    pub depth: u8,
    pub score: i32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub pv: &'a [Move],
    pub hashfull: usize,
    pub tablebase_hits: u64,
}

pub struct SharedSearch {
    pub table: TranspositionTable,
    helpers_stop: AtomicBool,
    nodes: AtomicU64,
    tablebase_hits: AtomicU64,
}

impl SharedSearch {
    pub fn new(hash_megabytes: usize) -> Self {
        Self {
            table: TranspositionTable::new(hash_megabytes),
            helpers_stop: AtomicBool::new(false),
            nodes: AtomicU64::new(0),
            tablebase_hits: AtomicU64::new(0),
        }
    }

    pub fn exclusive_table(shared: &mut Arc<Self>) -> &mut TranspositionTable {
        &mut Arc::get_mut(shared)
            .expect("no searcher holds the shared state")
            .table
    }

    fn begin(&self) {
        self.table.new_search();
        self.helpers_stop.store(false, Ordering::Relaxed);
        self.nodes.store(0, Ordering::Relaxed);
        self.tablebase_hits.store(0, Ordering::Relaxed);
    }

    fn count_tablebase_hits(&self, hits: u64) {
        self.tablebase_hits.fetch_add(hits, Ordering::Relaxed);
    }

    fn stop_helpers(&self) {
        self.helpers_stop.store(true, Ordering::Release);
    }

    fn helpers_stopped(&self) -> bool {
        self.helpers_stop.load(Ordering::Relaxed)
    }
}

pub struct Searcher {
    limits: SearchLimits,
    shared: Arc<SharedSearch>,
    stop_signal: Arc<AtomicBool>,
    ponder_signal: Arc<AtomicBool>,
    helper: bool,
    pondering: bool,
    start: Instant,
    clock_start: Instant,
    ponder_move: Option<Move>,
    nodes: u64,
    flushed_nodes: u64,
    ply: usize,
    stopped: bool,
    first_iteration_completed: bool,
    eval_noise: EvalNoise,
    pv: [[Move; MAX_PLY]; MAX_PLY],
    pv_len: [usize; MAX_PLY],
    eval_stack: [i32; MAX_PLY],
    excluded: [Move; MAX_PLY],
    killers: [[Move; 2]; MAX_PLY],
    history: History,
    played: [Option<PieceTo>; MAX_PLY],
    root_move_nodes: Box<RootMoveNodes>,
    tablebases: Option<Arc<Tablebases>>,
    tablebase_root_moves: Vec<Move>,
    tablebase_probing: bool,
    tablebases_in_search: bool,
}

fn soft_time_scale(stable_iterations: usize, score_drop: i32, best_move_node_share: f64) -> f64 {
    let stability = STABILITY_TIME_SCALES[stable_iterations.min(STABILITY_TIME_SCALES.len() - 1)];
    let (smallest_drop, largest_drop) = SCORE_DROP_RANGE;
    let score_trend =
        1.0 + f64::from(score_drop.clamp(smallest_drop, largest_drop)) / SCORE_DROP_TIME_DIVISOR;
    let node_share = (BEST_MOVE_NODES_PIVOT - best_move_node_share) * BEST_MOVE_NODES_SCALE;
    let (lowest, highest) = SOFT_TIME_SCALE_RANGE;
    (stability * score_trend * node_share).clamp(lowest, highest)
}

fn has_non_pawn_material(board: &Board, side: Color) -> bool {
    let pawns_and_kings = board.pieces(Piece::Pawn) | board.pieces(Piece::King);
    !(board.occupied_by(side) & !pawns_and_kings).is_empty()
}

fn pawn_key(board: &Board) -> u64 {
    let white = board.pieces_of(Color::White, Piece::Pawn).0;
    let black = board.pieces_of(Color::Black, Piece::Pawn).0;
    (white ^ black.rotate_left(BLACK_PAWNS_ROTATION)).wrapping_mul(PAWN_KEY_MULTIPLIER)
}

fn table_depth(depth: i32) -> u8 {
    depth.min(i32::from(u8::MAX)) as u8
}

fn late_move_reduction(depth: i32, move_number: usize) -> i32 {
    let depth = (depth as usize).min(REDUCTION_TABLE_SIZE - 1);
    REDUCTIONS[depth][move_number.min(REDUCTION_TABLE_SIZE - 1)]
}

pub fn is_mate_score(score: i32) -> bool {
    score.abs() >= MATE_BOUND
}

pub fn uci_score(score: i32) -> String {
    if !is_mate_score(score) {
        return format!("cp {score}");
    }
    let plies = MATE - score.abs();
    let moves = if score > 0 {
        (plies + 1) / 2
    } else {
        -plies / 2
    };
    format!("mate {moves}")
}

impl Searcher {
    pub fn new(
        limits: SearchLimits,
        stop_signal: Arc<AtomicBool>,
        shared: Arc<SharedSearch>,
    ) -> Self {
        Self {
            limits,
            shared,
            stop_signal,
            ponder_signal: Arc::default(),
            helper: false,
            pondering: false,
            start: Instant::now(),
            clock_start: Instant::now(),
            ponder_move: None,
            nodes: 0,
            flushed_nodes: 0,
            ply: 0,
            stopped: false,
            first_iteration_completed: false,
            eval_noise: EvalNoise::default(),
            pv: [[Move::NULL; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
            eval_stack: [-INFINITY; MAX_PLY],
            excluded: [Move::NULL; MAX_PLY],
            killers: [[Move::NULL; 2]; MAX_PLY],
            history: History::default(),
            played: [None; MAX_PLY],
            root_move_nodes: Box::new([[0; SQUARES]; SQUARES]),
            tablebases: None,
            tablebase_root_moves: Vec::new(),
            tablebase_probing: false,
            tablebases_in_search: true,
        }
    }

    pub fn with_tablebases(mut self, tablebases: Arc<Tablebases>) -> Self {
        self.tablebases = Some(tablebases);
        self
    }

    pub fn with_root_tablebases(mut self, tablebases: Arc<Tablebases>) -> Self {
        self.tablebases = Some(tablebases);
        self.tablebases_in_search = false;
        self
    }

    pub fn with_eval_noise(mut self, eval_noise: EvalNoise) -> Self {
        self.eval_noise = eval_noise;
        self
    }

    pub fn with_ponder_signal(mut self, ponder_signal: Arc<AtomicBool>) -> Self {
        self.ponder_signal = ponder_signal;
        self
    }

    pub fn search(&mut self, board: &mut Board, report: impl FnMut(&Iteration)) -> (Move, i32) {
        self.search_with_helpers(board, &mut [], report)
    }

    pub fn search_with_helpers(
        &mut self,
        board: &mut Board,
        helpers: &mut [Searcher],
        report: impl FnMut(&Iteration),
    ) -> (Move, i32) {
        self.shared.begin();
        self.prepare_tablebase_root(board);
        thread::scope(|scope| {
            for helper in helpers.iter_mut() {
                helper.tablebases.clone_from(&self.tablebases);
                helper
                    .tablebase_root_moves
                    .clone_from(&self.tablebase_root_moves);
                helper.tablebase_probing = self.tablebase_probing;
                let mut helper_board = board.clone();
                scope.spawn(move || helper.search_as_helper(&mut helper_board));
            }
            let best = self.search_as_main(board, report);
            self.shared.stop_helpers();
            best
        })
    }

    fn prepare_tablebase_root(&mut self, board: &mut Board) {
        let root = self
            .tablebases
            .as_ref()
            .and_then(|tablebases| tablebases.root_moves(board));
        self.tablebase_probing =
            self.tablebases.is_some() && root.is_none() && self.tablebases_in_search;
        self.tablebase_root_moves = root.map(|root| root.moves).unwrap_or_default();
        if !self.tablebase_root_moves.is_empty() {
            self.shared
                .count_tablebase_hits(self.tablebase_root_moves.len() as u64);
        }
    }

    fn excluded_at_root(&self, candidate: Move) -> bool {
        !self.tablebase_root_moves.is_empty() && !self.tablebase_root_moves.contains(&candidate)
    }

    fn probe_tablebase(&mut self, board: &mut Board) -> Option<(i32, Bound)> {
        let wdl = self.tablebases.as_ref()?.probe_wdl(board)?;
        self.shared.count_tablebase_hits(1);
        let ply = self.ply as i32;
        Some(match wdl {
            Wdl::Win => (TABLEBASE_WIN - ply, Bound::Lower),
            Wdl::Loss => (-TABLEBASE_WIN + ply, Bound::Upper),
            near_draw => (2 * near_draw.value(), Bound::Exact),
        })
    }

    fn reset(&mut self, helper: bool) {
        self.helper = helper;
        self.start = Instant::now();
        self.clock_start = self.start;
        self.pondering = !helper && self.ponder_signal.load(Ordering::Acquire);
        self.ponder_move = None;
        self.nodes = 0;
        self.flushed_nodes = 0;
        self.ply = 0;
        self.stopped = false;
        self.first_iteration_completed = false;
        self.root_move_nodes.fill([0; SQUARES]);
    }

    fn search_as_helper(&mut self, board: &mut Board) {
        self.reset(true);
        let mut score = 0;
        for depth in 1..=MAX_DEPTH {
            if self.shared.helpers_stopped() {
                break;
            }
            let iteration_score = self.aspiration_search(board, i32::from(depth), score);
            if self.stopped {
                break;
            }
            score = iteration_score;
        }
        self.flush_nodes();
    }

    fn search_as_main(
        &mut self,
        board: &mut Board,
        mut report: impl FnMut(&Iteration),
    ) -> (Move, i32) {
        self.reset(false);

        let max_depth = self.limits.depth.unwrap_or(MAX_DEPTH).clamp(1, MAX_DEPTH);
        let mut best = (Move::NULL, 0);
        let mut stable_iterations = 0;
        for depth in 1..=max_depth {
            let score = self.aspiration_search(board, i32::from(depth), best.1);
            let root_move_searched = self.pv_len[0] > 0;
            if self.stopped && !root_move_searched {
                break;
            }
            let pv = &self.pv[0][..self.pv_len[0]];
            let previous = best;
            best = (pv.first().copied().unwrap_or(Move::NULL), score);
            self.ponder_move = pv.get(1).copied();
            stable_iterations = if best.0 == previous.0 {
                stable_iterations + 1
            } else {
                0
            };
            report(&Iteration {
                depth,
                score,
                nodes: self.nodes(),
                elapsed: self.start.elapsed(),
                pv,
                hashfull: self.shared.table.hashfull(),
                tablebase_hits: self.shared.tablebase_hits.load(Ordering::Relaxed),
            });
            let score_drop = if depth == 1 { 0 } else { previous.1 - score };
            let time_scale = soft_time_scale(
                stable_iterations,
                score_drop,
                self.best_move_node_share(best.0),
            );
            self.first_iteration_completed = true;
            if self.stopped
                || self.node_limit_reached()
                || self.soft_node_limit_reached()
                || is_mate_score(score)
                || best.0 == Move::NULL
                || self.soft_time_expired(time_scale)
            {
                break;
            }
        }
        if best.0 == Move::NULL {
            best.0 = self
                .tablebase_root_moves
                .first()
                .copied()
                .or_else(|| board.legal_moves().first().copied())
                .unwrap_or(Move::NULL);
        }
        if self.ponder_move.is_none() && best.0 != Move::NULL {
            self.ponder_move = self.table_reply(board, best.0);
        }
        best
    }

    pub fn ponder_move(&self) -> Option<Move> {
        self.ponder_move
    }

    fn table_reply(&self, board: &mut Board, played: Move) -> Option<Move> {
        if !board.make_move(played) {
            return None;
        }
        let reply = self
            .shared
            .table
            .probe(board.state.zobrist_key, 0)
            .and_then(|hit| hit.best_move)
            .filter(|&reply| board.is_pseudo_legal(reply) && board.is_legal(reply));
        board.unmake_move();
        reply
    }

    pub fn with_history(mut self, history: History) -> Self {
        self.history = history;
        self
    }

    pub fn into_history(self) -> History {
        self.history
    }

    pub fn nodes(&self) -> u64 {
        self.shared.nodes.load(Ordering::Relaxed) + self.nodes - self.flushed_nodes
    }

    fn flush_nodes(&mut self) {
        let unflushed = self.nodes - self.flushed_nodes;
        self.shared.nodes.fetch_add(unflushed, Ordering::Relaxed);
        self.flushed_nodes = self.nodes;
    }

    fn aspiration_search(&mut self, board: &mut Board, depth: i32, previous_score: i32) -> i32 {
        if depth < ASPIRATION_DEPTH || is_mate_score(previous_score) {
            return self.negamax(board, depth, -INFINITY, INFINITY);
        }
        let mut window = ASPIRATION_WINDOW;
        let mut alpha = previous_score - window;
        let mut beta = previous_score + window;
        loop {
            let score = self.negamax(board, depth, alpha, beta);
            if self.stopped || (alpha < score && score < beta) {
                return score;
            }
            if score <= alpha {
                beta = (alpha + beta) / 2;
                alpha = (score - window).max(-INFINITY);
            } else {
                beta = (score + window).min(INFINITY);
            }
            window += window / 2;
            if window > ASPIRATION_FULL_WINDOW_THRESHOLD {
                alpha = -INFINITY;
                beta = INFINITY;
            }
        }
    }

    fn clock_running(&mut self) -> bool {
        if self.pondering && !self.ponder_signal.load(Ordering::Relaxed) {
            self.pondering = false;
            self.clock_start = Instant::now();
        }
        !self.pondering
    }

    fn soft_time_expired(&mut self, scale: f64) -> bool {
        self.clock_running()
            && self
                .limits
                .soft_time
                .is_some_and(|budget| self.clock_start.elapsed() >= budget.mul_f64(scale))
    }

    fn best_move_node_share(&self, best_move: Move) -> f64 {
        if self.nodes == 0 || best_move == Move::NULL {
            return 0.0;
        }
        let best_move_nodes =
            self.root_move_nodes[best_move.from().index()][best_move.to().index()];
        best_move_nodes as f64 / self.nodes as f64
    }

    fn hard_limit_reached(&mut self) -> bool {
        if self.helper {
            return self.shared.helpers_stopped();
        }
        self.stop_signal.load(Ordering::Relaxed)
            || (self.clock_running()
                && self
                    .limits
                    .time
                    .is_some_and(|budget| self.clock_start.elapsed() >= budget))
    }

    fn node_limit_reached(&self) -> bool {
        self.first_iteration_completed
            && self.limits.nodes.is_some_and(|limit| self.nodes() >= limit)
    }

    fn soft_node_limit_reached(&self) -> bool {
        self.limits
            .soft_nodes
            .is_some_and(|limit| self.nodes() >= limit)
    }

    fn visit_node(&mut self) {
        self.nodes += 1;
        if self.nodes.is_multiple_of(TIME_CHECK_INTERVAL) {
            self.flush_nodes();
        }
        if self.node_limit_reached()
            || (self.nodes.is_multiple_of(TIME_CHECK_INTERVAL) && self.hard_limit_reached())
        {
            self.stopped = true;
        }
    }

    fn static_evaluation(&self, board: &mut Board) -> i32 {
        nnue::evaluate(board) + self.eval_noise.offset(board.state.zobrist_key)
    }

    fn cached_static_evaluation(&self, board: &mut Board, hit: Option<Hit>) -> i32 {
        hit.and_then(|hit| hit.static_eval)
            .unwrap_or_else(|| self.static_evaluation(board))
    }

    fn corrected_evaluation(&self, board: &Board, raw_eval: i32) -> i32 {
        self.history
            .corrected_eval(board.state.active_color, pawn_key(board), raw_eval)
            .clamp(-MATE_BOUND + 1, MATE_BOUND - 1)
    }

    fn improving(&self, corrected_eval: Option<i32>) -> bool {
        let Some(eval) = corrected_eval else {
            return false;
        };
        [2, 4]
            .into_iter()
            .filter_map(|plies_back| self.ply.checked_sub(plies_back))
            .map(|ply| self.eval_stack[ply])
            .find(|&previous| previous != -INFINITY)
            .is_none_or(|previous| eval > previous)
    }

    fn update_pv(&mut self, best: Move) {
        let ply = self.ply;
        let child_len = self.pv_len[ply + 1].max(ply + 1);
        self.pv[ply][ply] = best;
        for index in ply + 1..child_len {
            self.pv[ply][index] = self.pv[ply + 1][index];
        }
        self.pv_len[ply] = child_len;
    }

    fn continuations(&self) -> Continuations {
        std::array::from_fn(|plies_back| {
            self.ply
                .checked_sub(plies_back + 1)
                .and_then(|ply| self.played[ply])
        })
    }

    fn reward_cutoff(
        &mut self,
        board: &Board,
        cutoff: Move,
        tried_quiets: &[Move],
        tried_noisy: &[Move],
        depth: i32,
    ) {
        let bonus = history_bonus(depth);
        if cutoff.is_noisy() {
            self.history.update_capture(
                moved_piece(board, cutoff),
                captured_piece(board, cutoff),
                bonus,
            );
        } else {
            let killers = &mut self.killers[self.ply];
            if killers[0] != cutoff {
                killers[1] = killers[0];
                killers[0] = cutoff;
            }
            let continuations = self.continuations();
            self.history
                .update_quiet(cutoff, moved_piece(board, cutoff), continuations, bonus);
            for &quiet in tried_quiets {
                self.history
                    .update_quiet(quiet, moved_piece(board, quiet), continuations, -bonus);
            }
        }
        for &noisy in tried_noisy {
            self.history.update_capture(
                moved_piece(board, noisy),
                captured_piece(board, noisy),
                -bonus,
            );
        }
    }

    fn probcut(
        &mut self,
        board: &mut Board,
        hash_move: Option<Move>,
        probcut_beta: i32,
        static_eval: i32,
        depth: i32,
    ) -> Option<(Move, i32)> {
        let continuations = self.continuations();
        let mut picker = MovePicker::good_noisy_only(hash_move);
        while let Some(candidate) = picker.next_move(board, &self.history, continuations) {
            if !board.see(candidate, probcut_beta - static_eval) {
                continue;
            }
            self.played[self.ply] = Some(moved_piece(board, candidate));
            if !board.make_move(candidate) {
                continue;
            }
            self.ply += 1;
            let mut score = -self.quiescence(board, -probcut_beta, -probcut_beta + 1);
            if score >= probcut_beta {
                score = -self.negamax(
                    board,
                    depth - PROBCUT_REDUCTION - 1,
                    -probcut_beta,
                    -probcut_beta + 1,
                );
            }
            self.ply -= 1;
            board.unmake_move();
            if self.stopped {
                return None;
            }
            if score >= probcut_beta {
                return Some((candidate, score));
            }
        }
        None
    }

    fn singular_extension(
        &mut self,
        board: &mut Board,
        candidate: Move,
        hit: Hit,
        depth: i32,
        beta: i32,
    ) -> Result<i32, i32> {
        let singular_beta = hit.score - SINGULAR_BETA_MARGIN * depth;
        self.excluded[self.ply] = candidate;
        let score = self.negamax(board, (depth - 1) / 2, singular_beta - 1, singular_beta);
        self.excluded[self.ply] = Move::NULL;
        if score < singular_beta {
            Ok(1)
        } else if singular_beta >= beta {
            Err(singular_beta)
        } else if hit.score >= beta {
            Ok(-1)
        } else {
            Ok(0)
        }
    }

    fn negamax(&mut self, board: &mut Board, depth: i32, mut alpha: i32, beta: i32) -> i32 {
        self.pv_len[self.ply] = self.ply;
        let is_root = self.ply == 0;
        let is_pv = beta - alpha > 1;
        if !is_root && (board.state.halfmove_clock >= 100 || board.is_repetition(self.ply)) {
            return 0;
        }
        if self.ply >= MAX_PLY - 1 {
            return self.static_evaluation(board);
        }
        let in_check = board.in_check();
        let mut depth = if in_check { depth + 1 } else { depth };
        if depth <= 0 {
            return self.quiescence(board, alpha, beta);
        }
        self.visit_node();
        if self.stopped {
            return 0;
        }
        self.killers[self.ply + 1] = [Move::NULL; 2];
        let excluded = self.excluded[self.ply];
        let singular_search = excluded != Move::NULL;

        let key = board.state.zobrist_key;
        let hit = self.shared.table.probe(key, self.ply);
        if let Some(score) = hit
            .filter(|_| !is_pv && !singular_search)
            .and_then(|hit| hit.cutoff_score(table_depth(depth), alpha, beta))
        {
            return score;
        }
        let original_alpha = alpha;
        let mut tablebase_floor = -INFINITY;
        let mut tablebase_ceiling = INFINITY;
        if !is_root
            && !singular_search
            && self.tablebase_probing
            && board.state.halfmove_clock == 0
            && let Some((score, bound)) = self.probe_tablebase(board)
        {
            let decisive = match bound {
                Bound::Exact => true,
                Bound::Lower => score >= beta,
                Bound::Upper => score <= alpha,
            };
            if decisive {
                self.shared.table.store(Store {
                    key,
                    best_move: Move::NULL,
                    score,
                    static_eval: None,
                    depth: table_depth(depth + TABLEBASE_DEPTH_BONUS),
                    bound,
                    ply: self.ply,
                });
                return score;
            }
            if is_pv {
                match bound {
                    Bound::Lower => {
                        tablebase_floor = score;
                        alpha = alpha.max(score);
                    }
                    Bound::Upper => tablebase_ceiling = score,
                    Bound::Exact => {}
                }
            }
        }
        let hash_move = hit.and_then(|hit| hit.best_move);
        if depth >= INTERNAL_REDUCTION_DEPTH && hash_move.is_none() {
            depth -= 1;
        }

        let us = board.state.active_color;
        let unrefined_eval = (!in_check).then(|| self.cached_static_evaluation(board, hit));
        let corrected_eval = unrefined_eval.map(|eval| self.corrected_evaluation(board, eval));
        let static_eval =
            corrected_eval.map_or(-INFINITY, |eval| hit.map_or(eval, |hit| hit.refine(eval)));
        self.eval_stack[self.ply] = corrected_eval.unwrap_or(-INFINITY);
        let improving = self.improving(corrected_eval);
        let prunable = !is_pv && !in_check && !singular_search;

        if prunable
            && depth <= REVERSE_FUTILITY_DEPTH
            && !is_mate_score(beta)
            && static_eval - REVERSE_FUTILITY_MARGIN * depth
                + REVERSE_FUTILITY_IMPROVING_MARGIN * i32::from(improving)
                >= beta
        {
            return (static_eval + beta) / 2;
        }

        if prunable
            && depth >= NULL_MOVE_DEPTH
            && static_eval >= beta
            && board.state.played_move != Move::NULL
            && has_non_pawn_material(board, us)
        {
            let reduction = 3 + depth / 4 + ((static_eval - beta) / 200).min(3);
            self.played[self.ply] = None;
            board.make_null_move();
            self.ply += 1;
            let score = -self.negamax(board, depth - 1 - reduction, -beta, -beta + 1);
            self.ply -= 1;
            board.unmake_null_move();
            if self.stopped {
                return 0;
            }
            if score >= beta {
                return if is_mate_score(score) { beta } else { score };
            }
        }

        let probcut_beta = beta + PROBCUT_MARGIN;
        let probcut_depth = depth - PROBCUT_REDUCTION + 1;
        if prunable
            && depth >= PROBCUT_DEPTH
            && !is_mate_score(beta)
            && !hit.is_some_and(|hit| {
                i32::from(hit.depth) >= probcut_depth && hit.score < probcut_beta
            })
        {
            if let Some((best_move, score)) =
                self.probcut(board, hash_move, probcut_beta, static_eval, depth)
            {
                self.shared.table.store(Store {
                    key,
                    best_move,
                    score,
                    static_eval: unrefined_eval,
                    depth: probcut_depth as u8,
                    bound: Bound::Lower,
                    ply: self.ply,
                });
                return score;
            }
            if self.stopped {
                return 0;
            }
        }

        let continuations = self.continuations();
        let mut picker = MovePicker::new(hash_move, self.killers[self.ply]);
        let hash_move_is_noisy = hash_move.is_some_and(Move::is_noisy);

        let mut best_score = tablebase_floor;
        let mut best_move = Move::NULL;
        let mut legal_moves = 0;
        let mut tried_quiets = [Move::NULL; TRACKED_QUIETS];
        let mut tried_quiet_count = 0;
        let mut tried_noisy = [Move::NULL; TRACKED_NOISY];
        let mut tried_noisy_count = 0;
        while let Some(candidate) = picker.next_move(board, &self.history, continuations) {
            if candidate == excluded || (is_root && self.excluded_at_root(candidate)) {
                continue;
            }
            let is_quiet = !candidate.is_noisy();
            let moved = moved_piece(board, candidate);
            let move_history = if is_quiet {
                self.history.quiet_score(candidate, moved, continuations)
            } else {
                self.history
                    .capture_score(moved, captured_piece(board, candidate))
            };
            if !is_root && !in_check && best_score > -MATE_BOUND {
                if is_quiet {
                    let late_move = !is_pv
                        && depth <= LATE_MOVE_PRUNING_DEPTH
                        && legal_moves >= (3 + depth * depth) / (2 - i32::from(improving));
                    if late_move {
                        picker.skip_quiets();
                        continue;
                    }
                    if !board.gives_check(candidate) {
                        let futile = depth <= FUTILITY_DEPTH
                            && static_eval + FUTILITY_BASE + FUTILITY_MARGIN * depth <= alpha;
                        if futile {
                            continue;
                        }
                        if depth <= HISTORY_PRUNING_DEPTH
                            && move_history < -HISTORY_PRUNING_MARGIN * depth
                        {
                            continue;
                        }
                        if depth <= QUIET_SEE_DEPTH
                            && !board.see(candidate, -QUIET_SEE_MARGIN * depth)
                        {
                            continue;
                        }
                    }
                } else if depth <= NOISY_SEE_DEPTH
                    && !board.see(candidate, -NOISY_SEE_MARGIN * depth)
                {
                    continue;
                }
            }

            let mut extension = 0;
            if let Some(hit) = hit
                && !is_root
                && !singular_search
                && depth >= SINGULAR_DEPTH
                && Some(candidate) == hash_move
                && hit.bound != Bound::Upper
                && i32::from(hit.depth) >= depth - SINGULAR_TABLE_DEPTH_MARGIN
                && !is_mate_score(hit.score)
            {
                match self.singular_extension(board, candidate, hit, depth, beta) {
                    _ if self.stopped => return 0,
                    Ok(singular_extension) => extension = singular_extension,
                    Err(multi_cut_score) => return multi_cut_score,
                }
            }

            let nodes_before = self.nodes;
            self.played[self.ply] = Some(moved);
            if !board.make_move(candidate) {
                continue;
            }
            legal_moves += 1;
            self.ply += 1;
            let new_depth = depth - 1 + extension;
            let score = if legal_moves == 1 {
                -self.negamax(board, new_depth, -beta, -alpha)
            } else {
                let gives_check = board.in_check();
                let mut reduction = 0;
                if depth >= REDUCTION_DEPTH && is_quiet && !in_check && !gives_check {
                    reduction = late_move_reduction(depth, legal_moves as usize);
                    reduction -= i32::from(is_pv);
                    reduction -= i32::from(self.killers[self.ply - 1].contains(&candidate));
                    reduction += i32::from(!improving);
                    reduction += i32::from(hash_move_is_noisy);
                    reduction -= move_history / REDUCTION_HISTORY_DIVISOR;
                    reduction = reduction.clamp(0, (new_depth - 1).max(0));
                }
                let mut score = -self.negamax(board, new_depth - reduction, -alpha - 1, -alpha);
                if !self.stopped && score > alpha && reduction > 0 {
                    score = -self.negamax(board, new_depth, -alpha - 1, -alpha);
                }
                if !self.stopped && score > alpha && score < beta {
                    score = -self.negamax(board, new_depth, -beta, -alpha);
                }
                score
            };
            self.ply -= 1;
            board.unmake_move();
            if is_root {
                self.root_move_nodes[candidate.from().index()][candidate.to().index()] +=
                    self.nodes - nodes_before;
            }
            if self.stopped {
                return if is_root { best_score } else { 0 };
            }
            best_score = best_score.max(score);
            if score > alpha {
                alpha = score;
                best_move = candidate;
                self.update_pv(candidate);
                if alpha >= beta {
                    self.reward_cutoff(
                        board,
                        candidate,
                        &tried_quiets[..tried_quiet_count],
                        &tried_noisy[..tried_noisy_count],
                        depth,
                    );
                    break;
                }
            }
            if is_quiet && tried_quiet_count < TRACKED_QUIETS {
                tried_quiets[tried_quiet_count] = candidate;
                tried_quiet_count += 1;
            } else if !is_quiet && tried_noisy_count < TRACKED_NOISY {
                tried_noisy[tried_noisy_count] = candidate;
                tried_noisy_count += 1;
            }
        }

        if singular_search {
            return if legal_moves == 0 { alpha } else { best_score };
        }
        let score = match legal_moves {
            0 if in_check => -MATE + self.ply as i32,
            0 => 0,
            _ => best_score.min(tablebase_ceiling),
        };
        let bound = if legal_moves == 0 {
            Bound::Exact
        } else if score >= beta {
            Bound::Lower
        } else if score <= original_alpha {
            Bound::Upper
        } else {
            Bound::Exact
        };
        if let (Some(raw_eval), Some(corrected_eval)) = (unrefined_eval, corrected_eval) {
            let quiet_best = best_move == Move::NULL || !best_move.is_noisy();
            let consistent = match bound {
                Bound::Lower => score > corrected_eval,
                Bound::Upper => score < corrected_eval,
                Bound::Exact => true,
            };
            if quiet_best && consistent && !is_mate_score(score) {
                self.history
                    .update_correction(us, pawn_key(board), depth, score - raw_eval);
            }
        }
        self.shared.table.store(Store {
            key,
            best_move,
            score,
            static_eval: unrefined_eval,
            depth: table_depth(depth),
            bound,
            ply: self.ply,
        });
        score
    }

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32) -> i32 {
        self.visit_node();
        if self.stopped {
            return 0;
        }
        if self.ply >= MAX_PLY - 1 {
            return self.static_evaluation(board);
        }
        let is_pv = beta - alpha > 1;
        let key = board.state.zobrist_key;
        let hit = self.shared.table.probe(key, self.ply);
        if let Some(score) = hit
            .filter(|_| !is_pv)
            .and_then(|hit| hit.cutoff_score(QUIESCENCE_DEPTH, alpha, beta))
        {
            return score;
        }
        let hash_move = hit.and_then(|hit| hit.best_move);
        let in_check = board.in_check();
        let mut unrefined_eval = None;
        let mut best_score = -INFINITY;
        let mut picker = if in_check {
            MovePicker::new(hash_move, [Move::NULL; 2])
        } else {
            let eval = self.cached_static_evaluation(board, hit);
            unrefined_eval = Some(eval);
            let corrected_eval = self.corrected_evaluation(board, eval);
            let stand_pat = hit.map_or(corrected_eval, |hit| hit.refine(corrected_eval));
            if stand_pat >= beta {
                if hit.is_none() {
                    self.shared.table.store(Store {
                        key,
                        best_move: Move::NULL,
                        score: stand_pat,
                        static_eval: unrefined_eval,
                        depth: QUIESCENCE_DEPTH,
                        bound: Bound::Lower,
                        ply: self.ply,
                    });
                }
                return stand_pat;
            }
            alpha = alpha.max(stand_pat);
            best_score = stand_pat;
            MovePicker::good_noisy_only(hash_move)
        };

        let futility_base = best_score + QUIESCENCE_FUTILITY_MARGIN;
        let continuations = self.continuations();
        let mut best_move = Move::NULL;
        let mut legal_moves = 0;
        while let Some(candidate) = picker.next_move(board, &self.history, continuations) {
            if !in_check && candidate.promotion().is_none() {
                let victim_value = captured_piece(board, candidate).map_or(0, see_value);
                if futility_base + victim_value <= alpha {
                    continue;
                }
                if futility_base <= alpha && !board.see(candidate, 1) {
                    continue;
                }
            }
            self.played[self.ply] = Some(moved_piece(board, candidate));
            if !board.make_move(candidate) {
                continue;
            }
            legal_moves += 1;
            self.ply += 1;
            let score = -self.quiescence(board, -beta, -alpha);
            self.ply -= 1;
            board.unmake_move();
            if self.stopped {
                return 0;
            }
            best_score = best_score.max(score);
            if score > alpha {
                alpha = score;
                best_move = candidate;
                if alpha >= beta {
                    break;
                }
            }
        }
        if in_check && legal_moves == 0 {
            return -MATE + self.ply as i32;
        }
        self.shared.table.store(Store {
            key,
            best_move,
            score: best_score,
            static_eval: unrefined_eval,
            depth: QUIESCENCE_DEPTH,
            bound: if best_score >= beta {
                Bound::Lower
            } else {
                Bound::Upper
            },
            ply: self.ply,
        });
        best_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::testing::board;
    use crate::board::{KIWIPETE, START_POSITION};
    use crate::syzygy::testing::local_tablebases;

    fn small_shared() -> Arc<SharedSearch> {
        Arc::new(SharedSearch::new(1))
    }

    #[test]
    fn corrected_evaluation_stays_below_mate_bound() {
        let ceiling = MATE_BOUND - 1;
        let cases = [
            (0, None, 0),
            (ceiling, None, ceiling),
            (ceiling, Some(MATE), ceiling),
            (-ceiling, Some(-MATE), -ceiling),
            (ceiling - 10, Some(MATE), ceiling),
            (ceiling, Some(-MATE), ceiling - 64),
            (100, Some(MATE), 164),
        ];

        for (raw_eval, search_error, expected) in cases {
            let position = board(KIWIPETE);
            let mut history = History::default();
            if let Some(error) = search_error {
                history.update_correction(
                    position.state.active_color,
                    pawn_key(&position),
                    MAX_DEPTH as i32,
                    error,
                );
            }
            let searcher = Searcher::new(SearchLimits::default(), Arc::default(), small_shared())
                .with_history(history);
            let corrected = searcher.corrected_evaluation(&position, raw_eval);
            assert_eq!(
                corrected, expected,
                "raw eval {raw_eval}, search error {search_error:?}"
            );
        }
    }

    #[test]
    fn searches_with_helpers() {
        let cases = [
            ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 0, 3, Some("a1a8")),
            ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 3, 3, Some("a1a8")),
            ("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", 2, 6, Some("d1d5")),
            (KIWIPETE, 3, 7, None),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 2, 4, Some("0000")),
        ];

        for (fen, helper_count, depth, expected_move) in cases {
            let mut board = board(fen);
            let shared = small_shared();
            let limits = SearchLimits {
                depth: Some(depth),
                ..SearchLimits::default()
            };
            let mut helpers: Vec<Searcher> = (0..helper_count)
                .map(|_| Searcher::new(limits, Arc::default(), Arc::clone(&shared)))
                .collect();
            let mut main = Searcher::new(limits, Arc::default(), Arc::clone(&shared));
            let mut main_nodes = 0;
            let (best_move, _) = main.search_with_helpers(&mut board, &mut helpers, |iteration| {
                main_nodes = iteration.nodes
            });
            let context = format!("fen {fen:?}, helpers {helper_count}");
            if let Some(expected) = expected_move {
                assert_eq!(best_move.to_string(), expected, "{context}");
            }
            assert!(shared.helpers_stopped(), "{context}");
            assert!(main.nodes() >= main_nodes, "{context}");
            assert_eq!(board.to_fen(), fen, "{context}: board restored");
        }
    }

    #[test]
    fn consults_tablebases_at_root_and_in_search() {
        let Some(tablebases) = local_tablebases() else {
            return;
        };
        let tablebases = Arc::new(tablebases);
        let cases = [
            ("8/8/4k3/8/8/8/1B6/1K2N3 w - - 0 1", 8, true, false),
            ("8/3k4/8/8/R7/8/1r6/2K5 w - - 0 1", 6, true, false),
            ("6k1/7p/8/8/8/2K5/3R3r/1R6 w - - 0 1", 6, false, true),
        ];

        for (fen, depth, root_in_tables, proves_win) in cases {
            let mut position = board(fen);
            let root = tablebases.root_moves(&mut position);
            assert_eq!(root.is_some(), root_in_tables, "fen {fen:?}");
            let limits = SearchLimits {
                depth: Some(depth),
                ..SearchLimits::default()
            };
            let mut tablebase_hits = 0;
            let (best_move, score) = Searcher::new(limits, Arc::default(), small_shared())
                .with_tablebases(Arc::clone(&tablebases))
                .search(&mut position, |iteration| {
                    tablebase_hits = iteration.tablebase_hits;
                });
            assert!(tablebase_hits > 0, "fen {fen:?}");
            if let Some(root) = root {
                assert!(
                    root.moves.contains(&best_move),
                    "fen {fen:?}, move {best_move}"
                );
            }
            if proves_win {
                assert!(
                    score > TABLEBASE_WIN - MAX_PLY as i32,
                    "fen {fen:?}, score {score}"
                );
            }
            assert_eq!(position.to_fen(), fen, "board restored for fen {fen:?}");
        }
    }

    #[test]
    fn finds_best_move() {
        let cases = [
            (
                "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
                3,
                Some("a1a8"),
                Some(MATE - 1),
            ),
            (
                "r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1",
                3,
                Some("a8a1"),
                Some(MATE - 1),
            ),
            ("4k3/8/8/8/8/8/8/RR4K1 w - - 0 1", 5, None, Some(MATE - 3)),
            ("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", 3, Some("d1d5"), None),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3, Some("0000"), Some(0)),
            (
                "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1",
                3,
                Some("0000"),
                Some(-MATE),
            ),
        ];

        for (fen, depth, expected_move, expected_score) in cases {
            let mut board = board(fen);
            let limits = SearchLimits {
                depth: Some(depth),
                ..SearchLimits::default()
            };
            let (best_move, score) =
                Searcher::new(limits, Arc::default(), small_shared()).search(&mut board, |_| {});
            if let Some(expected) = expected_move {
                assert_eq!(best_move.to_string(), expected, "fen {fen:?}");
            }
            if let Some(expected) = expected_score {
                assert_eq!(score, expected, "fen {fen:?}");
            }
            assert_eq!(board.to_fen(), fen, "board restored for fen {fen:?}");
        }
    }

    #[test]
    fn aspiration_recovers_from_wrong_previous_score() {
        let cases = [
            (
                "4k3/8/8/8/8/8/8/RR4K1 w - - 0 1",
                5,
                0,
                MATE - 3..=MATE - 3,
                None,
            ),
            (
                "4k3/8/8/8/8/8/8/RR4K1 w - - 0 1",
                5,
                3_000,
                MATE - 3..=MATE - 3,
                None,
            ),
            (
                "4k3/8/8/8/8/8/8/RR4K1 w - - 0 1",
                5,
                -3_000,
                MATE - 3..=MATE - 3,
                None,
            ),
            (
                "7k/8/6K1/8/8/8/8/1Q6 w - - 0 1",
                5,
                3_000,
                MATE - 1..=MATE - 1,
                Some("b1b8"),
            ),
            (
                "6k1/8/6K1/8/8/8/8/R7 w - - 0 1",
                5,
                5_000,
                MATE - 1..=MATE - 1,
                Some("a1a8"),
            ),
            (
                "6k1/8/6K1/8/8/8/8/R7 w - - 0 1",
                5,
                -5_000,
                MATE - 1..=MATE - 1,
                Some("a1a8"),
            ),
            (
                "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
                5,
                0,
                500..=INFINITY,
                Some("d1d5"),
            ),
            (
                "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
                5,
                -3_000,
                500..=INFINITY,
                Some("d1d5"),
            ),
            (
                "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
                5,
                3_000,
                500..=INFINITY,
                Some("d1d5"),
            ),
            (
                "3qk3/8/8/8/8/8/8/3QK3 w - - 0 1",
                6,
                2_000,
                -300..=300,
                None,
            ),
            (
                "3qk3/8/8/8/8/8/8/3QK3 w - - 0 1",
                6,
                -2_000,
                -300..=300,
                None,
            ),
        ];

        for (fen, depth, previous_score, expected_score, expected_move) in cases {
            let mut board = board(fen);
            let mut searcher =
                Searcher::new(SearchLimits::default(), Arc::default(), small_shared());
            let score = searcher.aspiration_search(&mut board, depth, previous_score);
            let context = format!("fen {fen:?}, depth {depth}, previous {previous_score}");
            assert!(expected_score.contains(&score), "{context}: score {score}");
            if let Some(expected) = expected_move {
                assert_eq!(searcher.pv[0][0].to_string(), expected, "{context}");
            }
            assert_eq!(board.to_fen(), fen, "{context}: board restored");
        }
    }

    #[test]
    fn quiescence_resolves_checks() {
        let cases = [
            ("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1", -MATE..=-MATE),
            ("4k3/p7/8/8/8/8/2n5/R3K3 w - - 0 1", -INFINITY..=-100),
            ("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", -50..=50),
            ("4k3/pppp4/8/8/8/8/PPPP4/R3K3 w - - 0 1", 300..=INFINITY),
        ];
        // TODO: cover more bare-king and lone-piece endings once datagen stops adjudicating them away

        for (fen, expected) in cases {
            let mut board = board(fen);
            let mut searcher =
                Searcher::new(SearchLimits::default(), Arc::default(), small_shared());
            let score = searcher.quiescence(&mut board, -INFINITY, INFINITY);
            assert!(expected.contains(&score), "fen {fen:?}, score {score}");
            assert_eq!(board.to_fen(), fen, "board restored for fen {fen:?}");
        }
    }

    #[test]
    fn keeps_best_move_from_interrupted_iteration() {
        let cases = [
            START_POSITION,
            KIWIPETE,
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];

        for fen in cases {
            let mut board = board(fen);
            let limits = SearchLimits {
                time: Some(Duration::ZERO),
                ..SearchLimits::default()
            };
            let mut last_report = None;
            let (best_move, _) = Searcher::new(limits, Arc::default(), small_shared())
                .search(&mut board, |iteration| {
                    last_report = Some((iteration.nodes, iteration.pv[0]))
                });
            let (nodes, pv_head) = last_report.expect("at least one iteration reported");
            assert!(nodes <= TIME_CHECK_INTERVAL, "fen {fen:?}, nodes {nodes}");
            assert_eq!(best_move, pv_head, "fen {fen:?}");
            assert_eq!(board.to_fen(), fen, "board restored for fen {fen:?}");
        }
    }

    #[test]
    fn ignores_clock_while_pondering() {
        let cases = [
            (START_POSITION, 5),
            (KIWIPETE, 4),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6),
        ];

        for (fen, depth) in cases {
            let mut board = board(fen);
            let limits = SearchLimits {
                depth: Some(depth),
                time: Some(Duration::ZERO),
                soft_time: Some(Duration::ZERO),
                ..SearchLimits::default()
            };
            let mut last_depth = 0;
            let mut searcher = Searcher::new(limits, Arc::default(), small_shared())
                .with_ponder_signal(Arc::new(AtomicBool::new(true)));
            let (best_move, _) =
                searcher.search(&mut board, |iteration| last_depth = iteration.depth);
            assert_eq!(last_depth, depth, "fen {fen:?}");
            let reply = searcher.ponder_move().expect("ponder move found");
            assert!(board.make_move(best_move), "fen {fen:?}");
            assert!(board.is_pseudo_legal(reply), "fen {fen:?}, reply {reply}");
            assert!(board.make_move(reply), "fen {fen:?}, reply {reply}");
        }
    }

    #[test]
    fn completes_first_iteration_under_node_limit() {
        let cases = [(START_POSITION, 1), (KIWIPETE, 1), (START_POSITION, 5_000)];

        for (fen, node_limit) in cases {
            let mut board = board(fen);
            let limits = SearchLimits {
                nodes: Some(node_limit),
                ..SearchLimits::default()
            };
            let noise = EvalNoise {
                amplitude: 400,
                seed: 42,
            };
            let mut reports = Vec::new();
            let (best_move, _) = Searcher::new(limits, Arc::default(), small_shared())
                .with_eval_noise(noise)
                .search(&mut board, |iteration| {
                    reports.push((iteration.depth, iteration.nodes, iteration.pv[0]))
                });
            let &(first_depth, first_nodes, _) = reports.first().expect("an iteration reported");
            let &(_, last_nodes, pv_head) = reports.last().expect("an iteration reported");
            assert_eq!(first_depth, 1, "fen {fen:?}");
            assert!(last_nodes <= node_limit.max(first_nodes), "fen {fen:?}");
            assert_ne!(best_move, Move::NULL, "fen {fen:?}");
            assert_eq!(best_move, pv_head, "fen {fen:?}");
            assert_eq!(board.to_fen(), fen, "board restored for fen {fen:?}");
        }
    }

    #[test]
    fn stops_after_the_iteration_that_reaches_soft_nodes() {
        let cases = [
            (START_POSITION, 5_000, None),
            (KIWIPETE, 5_000, None),
            (START_POSITION, 20_000, None),
            (KIWIPETE, 5_000, Some(50_000)),
            (KIWIPETE, 50_000, Some(5_000)),
        ];

        for (fen, soft_limit, hard_limit) in cases {
            let mut board = board(fen);
            let limits = SearchLimits {
                nodes: hard_limit,
                soft_nodes: Some(soft_limit),
                ..SearchLimits::default()
            };
            let mut reports = Vec::new();
            Searcher::new(limits, Arc::default(), small_shared()).search(&mut board, |iteration| {
                reports.push((iteration.depth, iteration.nodes))
            });
            let context = format!("fen {fen:?}, soft {soft_limit}, hard {hard_limit:?}");
            let depths: Vec<u8> = reports.iter().map(|&(depth, _)| depth).collect();
            let expected_depths: Vec<u8> = (1..=reports.len() as u8).collect();
            assert_eq!(depths, expected_depths, "{context}");
            let &(_, last_nodes) = reports.last().expect("an iteration reported");
            match hard_limit {
                Some(hard) if hard < soft_limit => {
                    assert!(last_nodes <= hard.max(reports[0].1), "{context}")
                }
                _ => {
                    assert!(last_nodes >= soft_limit, "{context}");
                    if let [.., (_, before_last), _] = reports.as_slice() {
                        assert!(*before_last < soft_limit, "{context}");
                    }
                }
            }
        }
    }

    #[test]
    fn scales_soft_time_by_search_instability() {
        let cases = [
            (0, 0, 0.5, 2.5),
            (2, 100, 0.5, 1.1 * 1.5 * 1.35),
            (3, 40, 0.8, 0.9 * 1.2 * 0.7 * 1.35),
            (4, 0, 0.9, 0.8 * 0.6 * 1.35),
            (10, 0, 0.9, 0.8 * 0.6 * 1.35),
            (4, -100, 1.0, 0.5),
        ];

        for (stable_iterations, score_drop, node_share, expected) in cases {
            let scale = soft_time_scale(stable_iterations, score_drop, node_share);
            assert!(
                (scale - expected).abs() < 1e-9,
                "stable {stable_iterations}, drop {score_drop}, share {node_share}: {scale}"
            );
        }
    }

    #[test]
    fn formats_uci_scores() {
        let cases = [
            (0, "cp 0"),
            (-35, "cp -35"),
            (MATE - 1, "mate 1"),
            (MATE - 3, "mate 2"),
            (-MATE + 2, "mate -1"),
            (-MATE, "mate 0"),
        ];

        for (score, expected) in cases {
            assert_eq!(uci_score(score), expected, "score {score}");
        }
    }
}
