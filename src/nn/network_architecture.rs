use nalgebra::{DMatrix, DVector};

pub struct Parameters {
    pub weights: Vec<DMatrix<f64>>,
    pub biases: Vec<DVector<f64>>,
}

impl Parameters {
    pub fn new(weights: Vec<DMatrix<f64>>, biases: Vec<DVector<f64>>) -> Self {
        Self { weights, biases }
    }
}

pub trait NetworkArchitecture {
    fn feedforward(p: &Parameters, a: DVector<f64>) -> DVector<f64>;
    fn training_feedforward(
        p: &Parameters,
        activation: &mut DVector<f64>,
        activations: &mut Vec<DVector<f64>>,
        zs: &mut Vec<DVector<f64>>,
    );
    fn get_derivative(z: &DVector<f64>) -> DVector<f64>;
}
