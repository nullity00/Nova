//! This module implements variable time multi-scalar multiplication using Icicle's GPU acceleration
use halo2curves::bn256::{Fr as Scalar, G1Affine as Affine, G1 as Point};
use icicle_core::{curve::Curve, msm, msm::MSMConfig};
use icicle_runtime::{memory::HostSlice, runtime, Device};

/// Initialize Icicle runtime if not already initialized
fn ensure_icicle_initialized() {
  // Load backend from environment or default
  let _ = runtime::load_backend_from_env_or_default();

  // Try to set CUDA device if available
  let cuda_device = Device::new("CUDA", 0);
  if runtime::is_device_available(&cuda_device).unwrap_or(false) {
    let _ = runtime::set_device(&cuda_device);
  }
}

/// A function that performs multi-scalar multiplication in variable time
pub fn vartime_multiscalar_mul(scalars: &[Scalar], bases: &[Affine]) -> Point {
  // Handle empty input case
  if scalars.is_empty() || bases.is_empty() {
    return Point::default();
  }

  ensure_icicle_initialized();

  // Prepare result vector
  let mut result = vec![Point::default(); 1];

  // Perform MSM using Icicle
  msm::msm(
    HostSlice::from_slice(scalars),
    HostSlice::from_slice(bases),
    &MSMConfig::default(),
    HostSlice::from_mut_slice(&mut result),
  )
  .unwrap_or_else(|_| {
    // Fallback to CPU implementation if Icicle fails
    let mut sum = Point::default();
    for (scalar, base) in scalars.iter().zip(bases.iter()) {
      sum += *base * scalar;
    }
    result[0] = sum;
  });

  result[0]
}

/// A function that performs a batch of multi-scalar multiplication in variable time
pub fn batch_vartime_multiscalar_mul(scalars: &[Vec<Scalar>], bases: &[Affine]) -> Vec<Point> {
  // Handle empty input case
  if scalars.is_empty() || bases.is_empty() {
    return vec![Point::default(); scalars.len()];
  }

  ensure_icicle_initialized();

  // Prepare result vector
  let mut results = vec![Point::default(); scalars.len()];

  // Process each batch separately
  for (i, scalar_batch) in scalars.iter().enumerate() {
    if scalar_batch.is_empty() {
      results[i] = Point::default();
      continue;
    }

    // Use only the needed number of bases for this batch
    let bases_to_use = &bases[..std::cmp::min(scalar_batch.len(), bases.len())];

    // Perform MSM for this batch
    let mut batch_result = vec![Point::default(); 1];

    msm::msm(
      HostSlice::from_slice(scalar_batch),
      HostSlice::from_slice(bases_to_use),
      &MSMConfig::default(),
      HostSlice::from_mut_slice(&mut batch_result),
    )
    .unwrap_or_else(|_| {
      // Fallback to CPU implementation if Icicle fails
      let mut sum = Point::default();
      for (scalar, base) in scalar_batch.iter().zip(bases_to_use.iter()) {
        sum += *base * scalar;
      }
      batch_result[0] = sum;
    });

    results[i] = batch_result[0];
  }

  results
}

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;
  use halo2curves::msm::msm_best;

  #[test]
  fn test_vartime_multiscalar_mul_empty() {
    let scalars = vec![];
    let bases = vec![];

    let result = vartime_multiscalar_mul(&scalars, &bases);

    assert_eq!(result, Point::default());
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul_empty() {
    let scalars = vec![vec![]];
    let bases = vec![];

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    assert_eq!(result, [Point::default(); 1]);
  }

  #[test]
  fn test_vartime_multiscalar_mul_simple() {
    let mut rng = rand::thread_rng();

    let scalars = vec![Scalar::random(&mut rng), Scalar::random(&mut rng)];
    let bases = vec![Affine::random(&mut rng), Affine::random(&mut rng)];

    let result = vartime_multiscalar_mul(&scalars, &bases);

    let expected = bases[0] * scalars[0] + bases[1] * scalars[1];

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul_simple() {
    let mut rng = rand::thread_rng();

    let scalars = vec![
      vec![Scalar::random(&mut rng), Scalar::random(&mut rng)],
      vec![Scalar::random(&mut rng), Scalar::random(&mut rng)],
    ];
    let bases = vec![Affine::random(&mut rng), Affine::random(&mut rng)];

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    assert_eq!(
      result[0],
      bases[0] * scalars[0][0] + bases[1] * scalars[0][1]
    );
    assert_eq!(
      result[1],
      bases[0] * scalars[1][0] + bases[1] * scalars[1][1]
    );
  }

  #[test]
  fn test_vartime_multiscalar_mul() {
    let mut rng = rand::thread_rng();
    let sample_len = 100;

    let (scalars, bases): (Vec<_>, Vec<_>) = (0..sample_len)
      .map(|_| (Scalar::random(&mut rng), Affine::random(&mut rng)))
      .unzip();

    let result = vartime_multiscalar_mul(&scalars, &bases);

    let mut expected = Point::default();
    for i in 0..sample_len {
      expected += bases[i] * scalars[i];
    }

    assert_eq!(result, expected);
  }

  #[test]
  fn test_vartime_multiscalar_mul_with_msm_best() {
    let mut rng = rand::thread_rng();
    let sample_len = 100;

    let (scalars, bases): (Vec<_>, Vec<_>) = (0..sample_len)
      .map(|_| (Scalar::random(&mut rng), Affine::random(&mut rng)))
      .unzip();

    let result = vartime_multiscalar_mul(&scalars, &bases);
    let expected = msm_best(&scalars, &bases);

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul() {
    let mut rng = rand::thread_rng();
    let batch_len = 20;
    let sample_len = 100;

    let scalars: Vec<Vec<Scalar>> = (0..batch_len)
      .map(|_| (0..sample_len).map(|_| Scalar::random(&mut rng)).collect())
      .collect();

    let bases: Vec<Affine> = (0..sample_len).map(|_| Affine::random(&mut rng)).collect();

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    let expected: Vec<Point> = scalars
      .iter()
      .map(|scalar_row| {
        scalar_row
          .iter()
          .enumerate()
          .map(|(i, scalar)| bases[i] * scalar)
          .sum()
      })
      .collect();

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul_with_msm_best() {
    let mut rng = rand::thread_rng();
    let batch_len = 20;
    let sample_len = 100;

    let scalars: Vec<Vec<Scalar>> = (0..batch_len)
      .map(|_| (0..sample_len).map(|_| Scalar::random(&mut rng)).collect())
      .collect();

    let bases: Vec<Affine> = (0..sample_len).map(|_| Affine::random(&mut rng)).collect();

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    let expected = scalars
      .iter()
      .map(|scalar| msm_best(scalar, &bases))
      .collect::<Vec<_>>();

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul_of_varying_sized_scalars_with_msm_best() {
    let mut rng = rand::thread_rng();
    let batch_len = 20;
    let sample_lens: Vec<usize> = (0..batch_len).map(|i| i * 100 / (batch_len - 1)).collect();

    let scalars: Vec<Vec<Scalar>> = (0..batch_len)
      .map(|i| {
        (0..sample_lens[i])
          .map(|_| Scalar::random(&mut rng))
          .collect()
      })
      .collect();

    let bases: Vec<Affine> = (0..sample_lens[batch_len - 1])
      .map(|_| Affine::random(&mut rng))
      .collect();

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    let expected = scalars
      .iter()
      .map(|scalar| msm_best(scalar, &bases[..scalar.len()]))
      .collect::<Vec<_>>();

    assert_eq!(result, expected);
  }
}
