use ndarray::prelude::*;

/// Computes the sphere function for a single 1D solution.
pub fn sphere_single(arr: ArrayView1<f64>) -> f64 {
    (&arr * &arr).sum()
}

/// Computes the sphere function for a batch of 1D solutions.
pub fn sphere(arr: ArrayView2<f64>) -> Array1<f64> {
    (&arr * &arr).sum_axis(Axis(1))
}
