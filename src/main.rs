use nalgebra::DVector;
use std::error::Error;

use crate::{data_provider::{Data, get_data}, neural_network::NeuralNetwork, regularization::{L1, L2}, relu_network::ReLuNetwork};

mod network_architecture;
mod regularization;
mod sigmoid_network;
mod relu_network;
mod neural_network;
mod data_provider;

fn main()-> Result<(), Box<dyn Error>> {
    let mut training_data: Vec<(DVector<f64>, DVector<f64>)> = Vec::new();
    get_data("data/mnist_train.csv", &mut Data::Training(&mut training_data))?;

    let mut test_data: Vec<(DVector<f64>, usize)> = Vec::new();
    get_data("data/mnist_test.csv", &mut Data::Test(&mut test_data))?;
    println!("Data parsing done");

    let mut network = NeuralNetwork::<ReLuNetwork, L1>::new(vec![784, 30, 10]);
    network.stochastic_gradient_descent(
                                    training_data,
                                    30, 
                                    10, 
                                    0.005,
                                    3.0);

    println!("All: {}", test_data.len());
    println!("Correct: {}", network.evaluate(test_data));
    

    Ok(())
}
