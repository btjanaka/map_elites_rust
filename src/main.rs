use clap::{Args, Parser, Subcommand};
use map_elites_rust::{benchmarks, utils};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

//
// Command line arguments
//

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Demonstration of the Sphere function.
    Sphere {
        /// Dimensionality of the 1D solutions.
        #[arg(short, long, default_value_t = 10)]
        dim: usize,
    },

    /// MAP-Elites on the Sphere function.
    MapElites(MapElitesConfig),
}

#[derive(Debug, Args)]
struct MapElitesConfig {
    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Iterations to run MAP-Elites.
    #[arg(long, default_value_t = 1000)]
    itrs: u32,

    /// Number of solutions to evaluate per iteration.
    #[arg(long, default_value_t = 100)]
    batch_size: usize,

    /// Dimensionality of the 1D solutions.
    #[arg(long, default_value_t = 10)]
    dim: usize,

    /// Number of cells along each side of the archive grid (currently this grid is a 2D grid with
    /// equal number of cells on each side.
    #[arg(long, default_value_t = 20)]
    cells: usize,

    /// Lower bound of measure space.
    #[arg(long, default_value_t = -1.0)]
    grid_min: f64,

    /// Upper bound of measure space.
    #[arg(long, default_value_t = 1.0)]
    grid_max: f64,

    /// Epsilon for grid cell calculations.
    #[arg(long, default_value_t = 1e-6)]
    epsilon: f64,

    /// Standard deviation of Gaussian noise.
    #[arg(long, default_value_t = 0.1)]
    sigma: f64,
}

//
// Functionality
//

fn sphere_demo(dim: usize) {
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
    let random_input: Array2<f64> = Array::random_using((2, dim), StandardNormal, &mut rng);
    println!("Random inputs: {}", random_input);
    println!(
        "With random inputs: {}",
        benchmarks::sphere(random_input.view())
    );
}

fn map_elites(config: &MapElitesConfig) {
    let mut rng = Pcg64Mcg::seed_from_u64(config.seed);

    let mut solution_arr = Array3::<f64>::zeros((config.cells, config.cells, config.dim));
    let mut objective_arr = Array2::<f64>::zeros((config.cells, config.cells));
    let mut measure_arr = Array3::<f64>::zeros((config.cells, config.cells, 2));
    let mut occupied_arr = Array2::<bool>::from_elem((config.cells, config.cells), false);
    let mut occupied_list = Vec::<[usize; 2]>::new();

    let interval_size = [
        config.grid_max - config.grid_min,
        config.grid_max - config.grid_min,
    ];

    for itr in 1..=config.itrs {
        // Get original solutions.
        let mut solutions = if itr == 1 {
            // Start with all zeros on iteration 1.
            Array2::<f64>::zeros((config.batch_size, config.dim))
        } else {
            // Sample random solutions from the archive. See here for info on creating an array
            // with individual elements set:
            // https://docs.rs/ndarray/0.15.6/ndarray/struct.ArrayBase.html#conversions-from-nested-vecsarrays
            let mut random_solutions = Array2::<f64>::zeros((config.batch_size, config.dim));
            for mut row in random_solutions.axis_iter_mut(Axis(0)) {
                let archive_idx = occupied_list.choose(&mut rng).unwrap();
                row.assign(&solution_arr.slice(s![archive_idx[0], archive_idx[1], ..]));
            }
            random_solutions
        };

        let noise =
            Array2::<f64>::random_using((config.batch_size, config.dim), StandardNormal, &mut rng)
                * config.sigma;

        // Create new solutions.
        solutions += &noise;

        // Evaluate solutions.
        let objectives = benchmarks::negative_sphere(solutions.view());
        let measures = benchmarks::first_two_vals(solutions.view());

        // Add to archive.
        for ((obj, meas), sol) in objectives
            .iter()
            .zip(measures.outer_iter())
            .zip(solutions.outer_iter())
        {
            // Compute archive index.
            let mut archive_idx = [0, 0];
            for i in [0, 1] {
                archive_idx[i] = utils::clip(
                    ((config.dim as f64 * (meas[i] - config.grid_min + config.epsilon))
                        / interval_size[i]) as usize,
                    0,
                    config.cells - 1,
                )
            }

            // Addition conditions -- cell not previously occupied, or objective value is greater.
            if !occupied_arr[archive_idx] || *obj > objective_arr[archive_idx] {
                let arr_slice = s![archive_idx[0], archive_idx[1], ..];
                solution_arr.slice_mut(arr_slice).assign(&sol);
                objective_arr[archive_idx] = *obj;
                measure_arr.slice_mut(arr_slice).assign(&meas);

                // Update occupancy only if not previously occupied -- this way, we do not have
                // duplicate entries in occupied_list.
                if !occupied_arr[archive_idx] {
                    occupied_arr[archive_idx] = true;
                    occupied_list.push(archive_idx);
                }
            }
        }
    }
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Sphere { dim } => {
            sphere_demo(*dim);
        }
        Commands::MapElites(config) => map_elites(config),
    }
}
