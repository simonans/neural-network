use nalgebra::{DMatrix, DVector};
use rand::{Rng, rng, seq::SliceRandom};
use std::{iter::zip, marker::PhantomData};

use super::fully_connected_layer::FullyConnectedLayer;
use super::network_architecture::{NetworkArchitecture, Parameters};
use super::regularization::Regularization;

use crate::utils::argmax;

pub struct NeuralNetwork<T: NetworkArchitecture, R: Regularization> {
    layer: FullyConnectedLayer<T, R>,
    mini_batch_size: f64,
}

impl<T: NetworkArchitecture, R: Regularization> NeuralNetwork<T, R> {
    pub fn new(sizes: Vec<usize>, learning_rate: f64, lambda: f64, mini_batch_size: f64) -> Self {
        Self {
            layer: FullyConnectedLayer::new(sizes, learning_rate, lambda),
            mini_batch_size,
        }
    }

    pub fn stochastic_gradient_descent(
        &mut self,
        mut training_data: Vec<(DVector<f64>, DVector<f64>)>, //Input data vector with corresponding desired output vector
        epochs: i32,
    ) {
        for e in 0..epochs {
            training_data.shuffle(&mut rng());
            for mini_batch in training_data.chunks(self.mini_batch_size as usize) {
                self.update_mini_batch(mini_batch, training_data.len() as f64);
            }
            println!("SGD: Epoch {} done", e);
        }
    }

    fn update_mini_batch(
        &mut self,
        mini_batch: &[(DVector<f64>, DVector<f64>)],
        trainig_dat_len: f64,
    ) {
        let mut nabla_b = Vec::new();
        for vector_len in self.layer.get_bias_sizes() {
            nabla_b.push(DVector::<f64>::zeros(*vector_len));
        }

        let mut nabla_w = Vec::new();
        for (rows, cols) in self.layer.get_weight_sizes() {
            nabla_w.push(DMatrix::<f64>::zeros(*rows, *cols));
        }

        //Alle Datensätze im mini_batch
        for (x, y) in mini_batch {
            let (delta_nabla_b, delta_nabla_w, _) = self.layer.backpropagation(x, y);
            for (nb, dnb) in zip(&mut nabla_b, &delta_nabla_b) {
                *nb += dnb;
            }
            for (nw, dnw) in zip(&mut nabla_w, &delta_nabla_w) {
                *nw += dnw;
            }
        }

        self.layer
            .update_parameters(&nabla_w, &nabla_b, self.mini_batch_size, trainig_dat_len);
    }

    pub fn evaluate(&self, test_data: Vec<(DVector<f64>, usize)>) -> usize {
        let mut erg: Vec<(usize, usize)> = Vec::new();
        for (x, y) in test_data {
            let output = self.layer.feedforward(x);
            erg.push((argmax(output), y));
        }
        let mut sum: usize = 0;
        for (x, y) in &erg {
            if x == y {
                sum += 1;
            }
        }
        sum
    }
}
