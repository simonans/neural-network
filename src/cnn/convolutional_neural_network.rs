use super::convolutional_layer::ConvoultionalLayer;
use crate::nn::{FullyConnectedLayer, L2, ReLuNetwork};

pub struct ConvolutionalNeuralNetwork {
    conv_layer: ConvoultionalLayer,
    fully_con_layer: FullyConnectedLayer<ReLuNetwork, L2>,
}

impl ConvolutionalNeuralNetwork {
    fn new() -> Self {}
}
