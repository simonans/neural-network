#![allow(dead_code)]

use crate::{cnn::utils::max_pooling, utils::relu_matrix};

use super::utils::{Tensor, tensor_convolution};

use nalgebra::DMatrix;
use rand::Rng;
use std::iter::zip;

const FILTER_SIZE: usize = 3;
const POOLING_SIZE: usize = 2;

pub struct ConvoultionalLayer {
    filters: Vec<Tensor>,
    biases: Vec<f64>,
    last_training_activation: Option<Tensor>,
    output_size: usize,
}

impl ConvoultionalLayer {
    pub fn new(number_of_filters: i32, number_of_input_channels: i32, input_size: usize) -> Self {
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
            last_training_activation: None,
            output_size: input_size - FILTER_SIZE + 1 + POOLING_SIZE,
        }
    }

    pub fn feedforward(&self, input: Tensor) -> Tensor {
        let mut feature_maps = Vec::new();

        for (filter, bias) in zip(&self.filters, &self.biases) {
            let mut m = tensor_convolution(&input, filter); //Convolution
            m.apply(|x| *x += bias);         //Bias
            relu_matrix(&mut m);                //Activation
            feature_maps.push(max_pooling(m));  //Pooling
        }

        Tensor::new(feature_maps)
    }

    pub fn training_feedforward(&mut self, input: Tensor) -> Tensor {       //TODO
        let mut feature_maps = Vec::new();
        let mut activations = Vec::new();

        //Convolution
        for filter in &self.filters {
            feature_maps.push(tensor_convolution(&input, filter));
        }

        //Bias add and Relu
        for (feature_map, bias) in zip(&mut feature_maps, &self.biases) {
            feature_map.apply(|x| *x += bias);
            activations.push(feature_map.clone());
            relu_matrix(feature_map);
        }
        self.last_training_activation = Some(Tensor::new(activations));
        Tensor::new(feature_maps)
    }

    //Strut for feedforward erg
    //gradient calc
    //update parameters
}
