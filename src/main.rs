use map_elites_rust::benchmarks;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

fn sphere_demo() {
    let input: Array2<f64> = arr2(&[[1., 1., 1., 1., 1.], [1., 2., 3., 4., 5.]]);
    println!("input: {}", input);
    println!("input[0]: {}", input.index_axis(Axis(0), 0));
    println!("input[0]: {}", input.slice(s![0, ..]));
    println!(
        "Sphere of input[0]: {}",
        benchmarks::sphere_single(input.index_axis(Axis(0), 0))
    );

    for (i, row) in input.outer_iter().enumerate() {
        println!("-> Sphere of row {}: {}", i, benchmarks::sphere_single(row));
    }

    println!("In batch: {}", benchmarks::sphere(input.view()));

    let mut rng = Pcg64Mcg::seed_from_u64(42);
    let random_input: Array2<f64> = Array::random_using((2, 10), StandardNormal, &mut rng);
    println!("Random inputs: {}", random_input);
    println!(
        "With random inputs: {}",
        benchmarks::sphere(random_input.view())
    );
}

fn main() {
    sphere_demo();
}
