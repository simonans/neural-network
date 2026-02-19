use nalgebra::{DMatrix, DVector};
use rand::{Rng, rng, seq::SliceRandom};
use std::{iter::zip, process::Output};

use super::{convolutional_layer::ConvoultionalLayer, flatten_layer::FlattenLayer};
use crate::{
    cnn::{flatten_layer, utils::Tensor},
    nn::{FullyConnectedLayer, L2, ReLuNetwork},
    utils::argmax,
};

const NUMBER_OF_FILTERS: usize = 64;
const PICTURE_SIZE: usize = 32;

pub struct ConvolutionalNeuralNetwork {
    conv_layer: ConvoultionalLayer,
    fully_con_layer: FullyConnectedLayer<ReLuNetwork, L2>,
    flatten_layer: FlattenLayer,
    mini_batch_size: f64,
}

impl ConvolutionalNeuralNetwork {
    pub fn new(mini_batch_size: f64, learning_rate: f64, lambda: f64) -> Self {
        let conv_layer =
            ConvoultionalLayer::new(NUMBER_OF_FILTERS as i32, 3, PICTURE_SIZE, learning_rate);
        let input_neurons = conv_layer.output_size * conv_layer.output_size * NUMBER_OF_FILTERS;
        let fully_con_layer = FullyConnectedLayer::<ReLuNetwork, L2>::new(
            vec![input_neurons, 30, 10],
            learning_rate,
            lambda,
        );
        let flatten_layer = FlattenLayer::new(
            conv_layer.output_size,
            conv_layer.output_size,
            NUMBER_OF_FILTERS,
        );
        Self {
            conv_layer,
            fully_con_layer,
            flatten_layer,
            mini_batch_size,
        }
    }

    pub fn stochastic_gradient_descent(
        &mut self,
        mut training_data: Vec<(Tensor, DVector<f64>)>, //Input data vector with corresponding desired output vector
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

    fn update_mini_batch(&mut self, mini_batch: &[(Tensor, DVector<f64>)], trainig_dat_len: f64) {
        //Ver, very ugly

        let mut fcl_nabla_b = Vec::new();
        for vector_len in self.fully_con_layer.get_bias_sizes() {
            fcl_nabla_b.push(DVector::<f64>::zeros(*vector_len));
        }

        let mut fcl_nabla_w = Vec::new();
        for (rows, cols) in self.fully_con_layer.get_weight_sizes() {
            fcl_nabla_w.push(DMatrix::<f64>::zeros(*rows, *cols));
        }

        let mut cl_nabla_b = vec![0.0; NUMBER_OF_FILTERS];

        let mut cl_nabla_w = Vec::new();
        for _ in 0..NUMBER_OF_FILTERS {
            cl_nabla_w.push(Tensor {
                data: vec![DMatrix::zeros(3, 3); NUMBER_OF_FILTERS],
            });
        }

        for (x, y) in mini_batch {
            let output = self.conv_layer.training_feedforward(x);
            let vec = self.flatten_layer.flatten(output);
            let (fcl_delta_nabla_b, fcl_delta_nabla_w, delta) =
                self.fully_con_layer.backpropagation(&vec, y);
            let delta_matrics = self.flatten_layer.back(delta);
            let (cl_delta_nabla_b, cl_delta_nabla_w) =
                self.conv_layer.backpropagation(&delta_matrics);

            for (nb, dnb) in zip(&mut fcl_nabla_b, &fcl_delta_nabla_b) {
                *nb += dnb;
            }
            for (nw, dnw) in zip(&mut fcl_nabla_w, &fcl_delta_nabla_w) {
                *nw += dnw;
            }

            for (nb, dnb) in zip(&mut cl_nabla_b, &cl_delta_nabla_b) {
                *nb += dnb;
            }
            for (tnw, tdnw) in zip(&mut cl_nabla_w, &cl_delta_nabla_w) {
                for (nw, dnw) in zip(&mut tnw.data, &tdnw.data) {
                    *nw += dnw;
                }
            }
        }
        self.fully_con_layer.update_parameters(
            &fcl_nabla_w,
            &fcl_nabla_b,
            self.mini_batch_size,
            trainig_dat_len,
        );

        self.conv_layer
            .update_parameters(&cl_nabla_w, &cl_nabla_b, self.mini_batch_size);
    }

    pub fn evaluate(&self, test_data: Vec<(Tensor, usize)>) -> usize {
        let mut erg: Vec<(usize, usize)> = Vec::new();
        for (x, y) in test_data {
            let conv_output = self.conv_layer.feedforward(x);
            let flatten = self.flatten_layer.flatten(conv_output);
            let output = self.fully_con_layer.feedforward(flatten);
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
