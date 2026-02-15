mod cnn;
mod nn;
mod utils;
use nn::{L1, NeuralNetwork, ReLuNetwork};
use utils::{MNIST_Data, get_mnist_data};

use nalgebra::{DMatrix, DVector};
use std::error::Error;

pub fn max_pooling(input: &DMatrix<f64>) -> DMatrix<f64> {
    let (rows, cols) = input.shape();
    assert!(rows % 2 == 0);
    assert!(cols % 2 == 0);
    let o_rows = rows / 2;
    let o_cols = cols / 2;
    let mut output = DMatrix::<f64>::zeros(o_rows, o_cols);

    for r in 0..o_rows {
        for c in 0..o_cols {
            let cur_r = 2 * r;
            let cur_c = 2 * c;
            output[(r, c)] = max(
                input[(cur_r, cur_c)],
                input[(cur_r + 1, cur_c)],
                input[(cur_r, cur_c + 1)],
                input[(cur_r + 1, cur_c + 1)],
            )
        }
    }
    output
}

fn max(v1: f64, v2: f64, v3: f64, v4: f64) -> f64 {
    v1.max(v2).max(v3).max(v4)
}

fn main() {
    let dm = DMatrix::from_row_slice(
        4,
        4,
        &[
            1.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, -1.0, -3.5, 0.0, 0.0, -4.0, -4.5,
        ],
    );

    println!("{}", max_pooling(&dm));
}

/*
fn main() -> Result<(), Box<dyn Error>> {
    let mut training_data: Vec<(DVector<f64>, DVector<f64>)> = Vec::new();
    get_mnist_data(
        "data/mnist_train.csv",
        &mut MNIST_Data::Training(&mut training_data),
    )?;

    let mut test_data: Vec<(DVector<f64>, usize)> = Vec::new();
    get_mnist_data("data/mnist_test.csv", &mut MNIST_Data::Test(&mut test_data))?;
    println!("Data parsing done");

    let mut network = NeuralNetwork::<ReLuNetwork, L1>::new(vec![784, 30, 10], 0.005, 3.0, 10.0);
    network.stochastic_gradient_descent(training_data, 30);

    println!("All: {}", test_data.len());
    println!("Correct: {}", network.evaluate(test_data));

    Ok(())
}
*/
