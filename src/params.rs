#[cfg(not(feature = "tune"))]
macro_rules! tunables {
    ($($name:ident = $value:expr, $min:expr, $max:expr, $step:expr;)*) => {
        $(
            #[inline(always)]
            pub const fn $name() -> i32 {
                $value
            }
        )*
    };
}

#[cfg(feature = "tune")]
macro_rules! tunables {
    ($($name:ident = $value:expr, $min:expr, $max:expr, $step:expr;)*) => {
        mod slots {
            use super::Tunable;
            $(
                #[allow(non_upper_case_globals)]
                pub static $name: Tunable = Tunable::new(stringify!($name), $value, $min, $max, $step);
            )*
        }

        $(
            #[inline(always)]
            pub fn $name() -> i32 {
                slots::$name.value()
            }
        )*

        pub static TUNABLES: &[&Tunable] = &[$(&slots::$name),*];
    };
}

#[cfg(feature = "tune")]
pub struct Tunable {
    pub name: &'static str,
    pub default: i32,
    pub min: i32,
    pub max: i32,
    pub step: f64,
    value: std::sync::atomic::AtomicI32,
}

#[cfg(feature = "tune")]
impl Tunable {
    const fn new(name: &'static str, default: i32, min: i32, max: i32, step: f64) -> Self {
        Self {
            name,
            default,
            min,
            max,
            step,
            value: std::sync::atomic::AtomicI32::new(default),
        }
    }

    pub fn value(&self) -> i32 {
        self.value.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set(&self, value: i32) {
        self.value.store(
            value.clamp(self.min, self.max),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

#[cfg(feature = "tune")]
pub fn find(name: &str) -> Option<&'static Tunable> {
    TUNABLES
        .iter()
        .copied()
        .find(|tunable| tunable.name.eq_ignore_ascii_case(name))
}

tunables! {
    quiescence_futility_margin = 267, 50, 400, 15.0;
    reverse_futility_margin = 89, 30, 160, 6.0;
    reverse_futility_improving_margin = 43, 0, 120, 5.0;
    null_move_eval_divisor = 185, 80, 400, 15.0;
    futility_base = 99, 20, 250, 8.0;
    futility_margin = 96, 40, 200, 8.0;
    futility_history_scale = 127, 0, 512, 12.0;
    noisy_see_margin = 88, 30, 200, 8.0;
    quiet_see_margin = 59, 20, 150, 5.0;
    history_pruning_margin = 2_137, 512, 4_096, 150.0;
    probcut_margin = 216, 80, 400, 15.0;
    lmr_base = 742, 250, 1_500, 60.0;
    lmr_divisor = 2_326, 1_200, 4_000, 150.0;
    quiet_reduction_history_scale = 69, 16, 192, 6.0;
    noisy_reduction_history_scale = 141, 32, 384, 12.0;
    noisy_reduction_discount = 1_102, 0, 2_560, 80.0;
    killer_reduction_discount = 1_081, 0, 2_560, 80.0;
    noisy_hash_move_reduction = 1_009, 0, 2_560, 80.0;
    threat_escape_reduction_discount = 1_125, 0, 2_560, 80.0;
    pv_reduction_discount = 1_049, 0, 2_560, 80.0;
    was_pv_reduction_discount = 1_083, 0, 2_560, 80.0;
    not_improving_reduction = 967, 0, 2_560, 80.0;
    check_reduction_discount = 1_029, 0, 2_560, 80.0;
    cut_node_reduction = 1_536, 0, 3_072, 100.0;
    child_cutoffs_reduction = 964, 0, 2_560, 80.0;
    pv_hash_score_above_alpha_discount = 530, 0, 1_536, 50.0;
    pv_hash_depth_discount = 504, 0, 1_536, 50.0;
    correction_reduction_scale = 3, 0, 12, 0.5;
    double_extension_margin = -2, -40, 80, 4.0;
    triple_extension_margin = 32, 0, 150, 5.0;
    pv_double_extension_margin = 201, 50, 400, 15.0;
    pv_triple_extension_margin = 281, 100, 500, 20.0;
    quiet_extension_margin_discount = 16, 0, 60, 2.0;
    deeper_research_margin = 41, 0, 120, 4.0;
    shallower_research_margin = 11, 0, 40, 1.5;
    strong_cutoff_margin = 80, 0, 200, 8.0;
    prior_fail_low_base_factor = 100, 0, 256, 10.0;
    prior_fail_low_deep_factor = 128, 0, 256, 10.0;
    prior_fail_low_eval_factor = 130, 0, 256, 10.0;
    prior_fail_low_eval_margin = 99, 0, 250, 8.0;
    eval_swing_history_scale = 17, 0, 48, 1.5;
    aspiration_window = 22, 8, 60, 2.0;
    history_bonus_scale = 18, 4, 48, 1.5;
    history_bonus_max = 1_204, 400, 2_400, 80.0;
}

#[cfg(all(test, feature = "tune"))]
mod tests {
    use super::*;

    #[test]
    fn clamps_tunables_and_finds_them_by_name() {
        let cases = [(150, 150), (-10, 0), (999, 200), (0, 0), (200, 200)];

        for (requested, expected) in cases {
            let tunable = Tunable::new("margin", 100, 0, 200, 8.0);
            tunable.set(requested);
            assert_eq!(tunable.value(), expected, "requested {requested}");
        }

        let lookups = [
            ("lmr_base", Some("lmr_base")),
            ("LMR_BASE", Some("lmr_base")),
            ("lmr", None),
            ("", None),
        ];
        for (name, expected) in lookups {
            assert_eq!(
                find(name).map(|tunable| tunable.name),
                expected,
                "name {name:?}"
            );
        }
    }
}
