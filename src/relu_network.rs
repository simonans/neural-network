#![allow(dead_code)]

use nalgebra::DVector;
use crate::network_architecture::{Parameters, NetworkArchitecture};
pub struct ReLuNetwork {}

impl NetworkArchitecture for ReLuNetwork {
    fn feedforward(p: &Parameters, mut a: DVector<f64>) -> DVector<f64> {
        let last = p.weights.len() - 1;
        for n in 0..last {
            a = relu(&(&p.weights[n] * &a + &p.biases[n]));
        }
        softmax(&(&p.weights[last] * &a + &p.biases[last]))
    }

    fn training_feedforward(p: &Parameters, 
                    activation: &mut DVector<f64>, 
                    activations: &mut Vec<DVector<f64>>, 
                    zs: &mut Vec<DVector<f64>>) {
        let last = p.weights.len() - 1;
        for n in 0..last {
            let z = &p.weights[n] * &*activation + &p.biases[n];
            *activation = relu(&z);
            zs.push(z);
            activations.push(activation.clone());
        }
        let z = &p.weights[last] * &*activation + &p.biases[last];
        *activation = softmax(&z);
        zs.push(z);
        activations.push(activation.clone());
    }

    fn get_derivative(z: &DVector<f64>) -> DVector<f64> {
        DVector::from_iterator(z.len(), z.iter().map(|x|if *x < 0.0 {0.0} else {1.0}))
    }
}

fn softmax(vec: &DVector<f64>) -> DVector<f64>{
    let denominator = vec.iter().fold(0.0, |acc, x| acc + x.exp());
    DVector::from_iterator(vec.len(), vec.iter()
                                                    .map(|x| x.exp() / denominator))
}

fn relu(vec: &DVector<f64>) -> DVector<f64> {
    DVector::from_iterator(vec.len(), vec.iter().map(|x| x.max(0.0)))
}