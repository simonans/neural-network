#![allow(dead_code)]

use nalgebra::DVector;
use std::iter::zip;
use super::network_architecture::{Parameters, NetworkArchitecture};
use crate::utils::sigmoid;

pub struct SigmoidNetwork {}

impl NetworkArchitecture for SigmoidNetwork {
    fn feedforward(p: &Parameters, mut a: DVector<f64>) -> DVector<f64> {
        for (w, b) in zip(&p.weights, &p.biases) {
            a = sigmoid(&(w * a + b));
        }
        a
    }

    fn training_feedforward(p: &Parameters, 
                    activation: &mut DVector<f64>, 
                    activations: &mut Vec<DVector<f64>>, 
                    zs: &mut Vec<DVector<f64>>) {
        for (w, b) in zip(&p.weights, &p.biases) {
            let z = w * &*activation + b;   //Activation ist mut Referenz, * erwartet Referenz
                                                                        //Deswegen erst dereferenziren und dann eine Referenz daraus machen
            *activation = sigmoid(&z);
            zs.push(z);
            activations.push(activation.clone());
        }
    }

    fn get_derivative(z: &DVector<f64>) -> DVector<f64> {
        let sig = sigmoid(z);
        sig.component_mul(&sig.map(|x| 1.0 - x))
    }
}


