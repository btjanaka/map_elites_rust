use ndarray::prelude::*;

/// Computes the sphere function for a single 1D solution.
fn sphere_single(arr: ArrayView1<f64>) -> f64 {
    (&arr * &arr).sum()
}

fn sphere(arr: ArrayView2<f64>) -> Array1<f64> {
    (&arr * &arr).sum_axis(Axis(1))
}

fn sphere_demo() {
    let input: Array2<f64> = arr2(&[[1., 1., 1., 1., 1.], [1., 2., 3., 4., 5.]]);
    println!("input: {}", input);
    println!("input[0]: {}", input.index_axis(Axis(0), 0));
    println!("input[0]: {}", input.slice(s![0, ..]));
    println!(
        "Sphere of input[0]: {}",
        sphere_single(input.index_axis(Axis(0), 0))
    );

    for (i, row) in input.outer_iter().enumerate() {
        println!("-> Sphere of row {}: {}", i, sphere_single(row));
    }

    println!("In batch: {}", sphere(input.view()));
}

fn main() {
    sphere_demo();
    map_elites_rust::test();
}
