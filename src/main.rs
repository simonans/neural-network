mod cnn;
mod nn;
mod utils;
use cnn::ConvolutionalNeuralNetwork;
use nn::{L1, NeuralNetwork, ReLuNetwork};
use utils::{Cifar_Data, get_cifar_data};

use nalgebra::{DMatrix, DVector};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut training_data = Vec::new();
    let training_files = vec![
        "data/data_batch_1.bin",
        "data/data_batch_2.bin",
        "data/data_batch_3.bin",
        "data/data_batch_4.bin",
        "data/data_batch_5.bin",
    ];
    get_cifar_data(
        &training_files,
        &mut Cifar_Data::Training(&mut training_data),
    )?;

    println!("{}", training_data.len());

    let mut test_data = Vec::new();
    let test_files = vec!["data/test_batch.bin"];
    get_cifar_data(&test_files, &mut Cifar_Data::Test(&mut test_data))?;
    println!("Data parsing done");

    let mut network = ConvolutionalNeuralNetwork::new(64.0, 0.005, 3.0);
    network.stochastic_gradient_descent(training_data, 8);

    println!("All: {}", test_data.len());
    println!("Correct: {}", network.evaluate(test_data));

    Ok(())
}
