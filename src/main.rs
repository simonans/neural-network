mod nn;
mod utils;
use nn::{L1, NeuralNetwork, ReLuNetwork};
use utils::{MNIST_Data, get_mnist_data};

use nalgebra::DVector;
use std::error::Error;

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
