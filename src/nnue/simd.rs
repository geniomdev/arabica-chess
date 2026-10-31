use super::{HIDDEN, QA};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi32, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_max_epi16,
    _mm256_min_epi16, _mm256_mullo_epi16, _mm256_set1_epi16, _mm256_setzero_si256,
    _mm256_storeu_si256, _mm512_add_epi32, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_max_epi16,
    _mm512_min_epi16, _mm512_mullo_epi16, _mm512_set1_epi16, _mm512_setzero_si512,
    _mm512_storeu_si512,
};

pub const MADD_WEIGHT_LIMIT: i16 = 128;

#[cfg(target_arch = "x86_64")]
const AVX2_LANES: usize = 16;
#[cfg(target_arch = "x86_64")]
const AVX2_SUMS: usize = 8;
#[cfg(target_arch = "x86_64")]
const AVX512_LANES: usize = 32;
#[cfg(target_arch = "x86_64")]
const AVX512_SUMS: usize = 16;
#[cfg(target_arch = "x86_64")]
const _: () = assert!(
    (HIDDEN / AVX2_LANES) as i64 * 2 * (QA * QA) as i64 * MADD_WEIGHT_LIMIT as i64
        <= i32::MAX as i64
);
#[cfg(target_arch = "x86_64")]
const _: () = assert!(
    (HIDDEN / AVX512_LANES) as i64 * 2 * (QA * QA) as i64 * MADD_WEIGHT_LIMIT as i64
        <= i32::MAX as i64
);
#[cfg(target_arch = "x86_64")]
const _: () = assert!(HIDDEN.is_multiple_of(AVX512_LANES));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputKernel {
    Portable,
    Avx2,
    Avx512,
}

pub static FASTEST_KERNEL: std::sync::LazyLock<OutputKernel> = std::sync::LazyLock::new(|| {
    supported_kernels()
        .last()
        .copied()
        .unwrap_or(OutputKernel::Portable)
});

pub fn supported_kernels() -> Vec<OutputKernel> {
    let mut kernels = vec![OutputKernel::Portable];
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            kernels.push(OutputKernel::Avx2);
        }
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            kernels.push(OutputKernel::Avx512);
        }
    }
    kernels
}

fn screlu_dot_portable(accumulator: &[i16; HIDDEN], weights: &[i16]) -> i64 {
    accumulator
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| {
            let clipped = i64::from(value).clamp(0, i64::from(QA));
            clipped * clipped * i64::from(weight)
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn screlu_dot_avx2(accumulator: &[i16; HIDDEN], weights: &[i16]) -> i64 {
    assert_eq!(weights.len(), HIDDEN);
    let zero = _mm256_setzero_si256();
    let ceiling = _mm256_set1_epi16(QA as i16);
    let mut sum = _mm256_setzero_si256();
    for offset in (0..HIDDEN).step_by(AVX2_LANES) {
        let (values, row) = unsafe {
            (
                _mm256_loadu_si256(accumulator.as_ptr().add(offset).cast::<__m256i>()),
                _mm256_loadu_si256(weights.as_ptr().add(offset).cast::<__m256i>()),
            )
        };
        let clipped = _mm256_min_epi16(_mm256_max_epi16(values, zero), ceiling);
        let weighted = _mm256_mullo_epi16(clipped, row);
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(weighted, clipped));
    }
    let mut lanes = [0i32; AVX2_SUMS];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), sum) };
    lanes.iter().map(|&lane| i64::from(lane)).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn screlu_dot_avx512(accumulator: &[i16; HIDDEN], weights: &[i16]) -> i64 {
    assert_eq!(weights.len(), HIDDEN);
    let zero = _mm512_setzero_si512();
    let ceiling = _mm512_set1_epi16(QA as i16);
    let mut sum = _mm512_setzero_si512();
    for offset in (0..HIDDEN).step_by(AVX512_LANES) {
        let (values, row) = unsafe {
            (
                _mm512_loadu_si512(accumulator.as_ptr().add(offset).cast::<__m512i>()),
                _mm512_loadu_si512(weights.as_ptr().add(offset).cast::<__m512i>()),
            )
        };
        let clipped = _mm512_min_epi16(_mm512_max_epi16(values, zero), ceiling);
        let weighted = _mm512_mullo_epi16(clipped, row);
        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(weighted, clipped));
    }
    let mut lanes = [0i32; AVX512_SUMS];
    unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast::<__m512i>(), sum) };
    lanes.iter().map(|&lane| i64::from(lane)).sum()
}

pub fn screlu_dot(kernel: OutputKernel, accumulator: &[i16; HIDDEN], weights: &[i16]) -> i64 {
    match kernel {
        #[cfg(target_arch = "x86_64")]
        OutputKernel::Avx512 => unsafe { screlu_dot_avx512(accumulator, weights) },
        #[cfg(target_arch = "x86_64")]
        OutputKernel::Avx2 => unsafe { screlu_dot_avx2(accumulator, weights) },
        _ => screlu_dot_portable(accumulator, weights),
    }
}

pub fn fits_madd(weights: &[i16]) -> bool {
    weights
        .iter()
        .all(|weight| weight.unsigned_abs() <= MADD_WEIGHT_LIMIT.unsigned_abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Random;

    fn random_row(random: &mut Random, low: i16, high: i16) -> [i16; HIDDEN] {
        let span = (i32::from(high) - i32::from(low) + 1) as usize;
        std::array::from_fn(|_| (i32::from(low) + random.below(span) as i32) as i16)
    }

    #[test]
    fn vector_dot_products_match_portable_ones() {
        let cases = [
            (1, -600, 600, -128, 128),
            (2, 0, 255, -128, 128),
            (3, -32_768, 32_767, -128, 128),
            (4, 200, 400, 128, 128),
            (5, 255, 255, -128, -128),
            (6, 255, 255, 128, 128),
        ];

        for kernel in supported_kernels() {
            for (seed, low, high, weight_low, weight_high) in cases {
                let mut random = Random::new(seed);
                let accumulator = random_row(&mut random, low, high);
                let weights = random_row(&mut random, weight_low, weight_high);
                assert_eq!(
                    screlu_dot(kernel, &accumulator, &weights),
                    screlu_dot(OutputKernel::Portable, &accumulator, &weights),
                    "kernel {kernel:?}, seed {seed}"
                );
            }
        }
    }

    #[test]
    fn saturated_dot_products_exceed_i32() {
        let accumulator = [QA as i16; HIDDEN];
        let weights = [MADD_WEIGHT_LIMIT; HIDDEN];
        let expected = i64::from(QA * QA) * i64::from(MADD_WEIGHT_LIMIT) * HIDDEN as i64;
        assert!(expected > i64::from(i32::MAX));
        for kernel in supported_kernels() {
            assert_eq!(
                screlu_dot(kernel, &accumulator, &weights),
                expected,
                "kernel {kernel:?}"
            );
        }
    }

    #[test]
    fn fastest_kernel_is_supported() {
        let supported = supported_kernels();
        assert_eq!(supported.first(), Some(&OutputKernel::Portable));
        assert!(
            supported.contains(&*FASTEST_KERNEL),
            "{:?}",
            *FASTEST_KERNEL
        );
        assert_eq!(supported.last(), Some(&*FASTEST_KERNEL));
    }

    #[test]
    fn checks_weights_for_madd() {
        let cases: [(&[i16], bool); 4] = [
            (&[0, 127, -127], true),
            (&[128, -128], true),
            (&[129], false),
            (&[-129, 0], false),
        ];

        for (weights, expected) in cases {
            assert_eq!(fits_madd(weights), expected, "weights {weights:?}");
        }
    }
}
