use ndarray::prelude::*;

/// Computes the sphere function for a single 1D solution.
pub fn sphere_single(arr: ArrayView1<f64>) -> f64 {
    (&arr * &arr).sum()
}

/// Computes the sphere function for a batch of 1D solutions.
pub fn sphere(arr: ArrayView2<f64>) -> Array1<f64> {
    (&arr * &arr).sum_axis(Axis(1))
}

/// Computes the sphere function for a batch of 1D solutions.
pub fn negative_sphere(arr: ArrayView2<f64>) -> Array1<f64> {
    -(&arr * &arr).sum_axis(Axis(1))
}

/// Computes simple measures that are just the first two entries of every solution.
pub fn first_two_vals(arr: ArrayView2<f64>) -> Array2<f64> {
    let mut res = Array2::<f64>::zeros((arr.nrows(), 2));
    res.assign(&arr.slice(s![.., ..2]));
    res
}
