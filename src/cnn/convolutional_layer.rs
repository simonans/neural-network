#![allow(dead_code)]

use crate::cnn::utils::{
    Tensor, matrix_convolution, max_pooling, relu_derivative, tensor_convolution,
    training_max_pooling,
};

use crate::utils::relu_matrix;

use itertools::izip;
use nalgebra::DMatrix;
use rand::Rng;
use std::iter::zip;

const FILTER_SIZE: usize = 3;
const POOLING_SIZE: usize = 2;

struct TrainingCache {
    pub activation: Tensor,
    pub z: Tensor,
    pub pooling_indizes: Vec<Vec<(usize, usize)>>,
}

impl TrainingCache {
    fn new() -> Self {
        Self {
            activation: Tensor { data: Vec::new() },
            z: Tensor { data: Vec::new() },
            pooling_indizes: Vec::new(),
        }
    }
}

pub struct ConvoultionalLayer {
    filters: Vec<Tensor>,
    biases: Vec<f64>,
    training_cache: TrainingCache,
    learning_rate: f64,
    output_size: usize,
}

impl ConvoultionalLayer {
    pub fn new(
        number_of_filters: i32,
        number_of_input_channels: i32,
        input_size: usize,
        learning_rate: f64,
    ) -> Self {
        let mut filters = Vec::new();
        let mut biases = Vec::new();
        let scale = 2 / (2 * FILTER_SIZE * number_of_input_channels as usize);
        let mut rng = rand::rng();
        for _ in 0..number_of_filters {
            let mut t = Vec::new();
            for _ in 0..number_of_input_channels {
                t.push(DMatrix::<f64>::from_fn(FILTER_SIZE, FILTER_SIZE, |_, _| {
                    rng.random_range(0.0..scale as f64) //Gauß
                }));
            }
            filters.push(Tensor::new(t));

            biases.push(0.0);
        }

        Self {
            filters,
            biases,
            training_cache: TrainingCache::new(),
            learning_rate,
            output_size: input_size - FILTER_SIZE + 1 + POOLING_SIZE,
        }
    }

    pub fn feedforward(&self, input: Tensor) -> Tensor {
        let mut feature_maps = Vec::new();

        for (filter, bias) in zip(&self.filters, &self.biases) {
            let mut m = tensor_convolution(&input, filter); //Convolution
            m.apply(|x| *x += bias); //Bias
            relu_matrix(&mut m); //Activation
            feature_maps.push(max_pooling(m)); //Pooling
        }

        Tensor::new(feature_maps)
    }

    pub fn training_feedforward(&mut self, input: Tensor) -> Tensor {
        let mut feature_maps = Vec::new();
        let mut cached_maps = Vec::new();

        self.training_cache.activation.data = input.data.clone();
        for (filter, bias) in zip(&self.filters, &self.biases) {
            let mut m = tensor_convolution(&input, filter); //Convolution
            m.apply(|x| *x += bias); //Bias
            relu_matrix(&mut m); //Activation
            cached_maps.push(m.clone());
            let training_pool = training_max_pooling(m); //Pooling
            feature_maps.push(training_pool.0);
            self.training_cache.pooling_indizes.push(training_pool.1);
        }

        self.training_cache.z = Tensor { data: cached_maps };

        Tensor::new(feature_maps)
    }

    pub fn backpropagation(&self, back_input: &Tensor) -> (Vec<f64>, Vec<Tensor>) {
        let mut nabla_b = Vec::new();
        let mut nabla_w = Vec::new();

        //Max Pooling Rückwärts
        let mut delta = Tensor {
            data: vec![
                DMatrix::<f64>::zeros(self.output_size, self.output_size);
                back_input.data.len()
            ],
        };
        for (indizes, back_layer, after_layer) in izip!(
            &self.training_cache.pooling_indizes,
            &mut delta.data,
            &back_input.data
        ) {
            for (idx, v) in zip(indizes, after_layer) {
                back_layer[*idx] = *v;
            }
        }

        let derivative = relu_derivative(&self.training_cache.z);

        for (delta_layer, derivative_layer) in zip(&delta.data, &derivative.data) {
            let d = delta_layer.component_mul(derivative_layer);
            nabla_b.push(d.iter().sum());
            let mut data = Vec::new();
            for channel in &self.training_cache.activation.data {
                data.push(matrix_convolution(channel, &d));
            }
            nabla_w.push(Tensor { data });
        }
        (nabla_b, nabla_w)
    }

    fn update_parameters(
        &mut self,
        nabla_w: &[Tensor],
        nabla_b: &[f64],
        mini_batch_size: f64,
    ) {
        let mini_batch_learning_rate = self.learning_rate / mini_batch_size;

        for (w, nw) in zip(&mut self.filters, nabla_w) {
            w.learning(mini_batch_learning_rate,nw);
        }

        for (b, nb) in zip(&mut self.biases, nabla_b) {
            *b -= mini_batch_learning_rate * nb; 
        }
    }

    //Strut for feedforward erg
    //gradient calc
    //update parameters
}
