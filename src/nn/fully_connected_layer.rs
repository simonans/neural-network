use nalgebra::{DMatrix, DVector};
use rand::Rng;

use super::network_architecture::{NetworkArchitecture, Parameters};
use super::regularization::Regularization;

use std::{iter::zip, marker::PhantomData};

pub struct FullyConnectedLayer<T: NetworkArchitecture, R: Regularization> {
    parameters: Parameters,
    weight_sizes: Vec<(usize, usize)>, //Rows, Cols
    bias_sizes: Vec<usize>,
    learning_rate: f64,
    lambda: f64, //Regularization hyperparameter
    architecture: PhantomData<T>,
    regularization: PhantomData<R>,
}

impl<T: NetworkArchitecture, R: Regularization> FullyConnectedLayer<T, R> {
    pub fn new(sizes: Vec<usize>, learning_rate: f64, lambda: f64) -> Self {
        let mut biases = Vec::new();
        let mut weights = Vec::new();
        let mut weight_sizes = Vec::new();
        let mut bias_sizes = Vec::new();
        let mut rng = rand::rng();
        for n in 0..(sizes.len() - 1) {
            weights.push(DMatrix::<f64>::from_fn(sizes[n + 1], sizes[n], |_, _| {
                let scale = (1.0 / sizes[n] as f64).sqrt();
                rng.random_range(-scale..scale) //Gauß
            }));
            weight_sizes.push((sizes[n + 1], sizes[n]));
            //w11 w21 w31
            //w12 w22 w32  Höhe = Rows = Anzahl Neuronen rechter Layer (Output)
            //w13 w23 w33  Breite = Cols = Anzahl Neuronen linker Layer (Input)
            //w14 w24 w34  Hier mapt 3 Neuronen auf 4 Neuronen

            biases.push(DVector::<f64>::zeros(sizes[n + 1]));
            bias_sizes.push(sizes[n + 1]);
            //Input Layer hat keine Gewichte
        }

        Self {
            parameters: Parameters::new(weights, biases),
            weight_sizes,
            bias_sizes,
            learning_rate,
            lambda,
            architecture: PhantomData,
            regularization: PhantomData,
        }
    }

    pub fn feedforward(&self, input: DVector<f64>) -> DVector<f64> {
        T::feedforward(&self.parameters, input)
    }

    pub fn backpropagation(
        &self,
        x: &DVector<f64>,
        y: &DVector<f64>,
    ) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
        let mut nabla_b = Vec::new();
        for bias_vector in &self.parameters.biases {
            nabla_b.push(DVector::<f64>::zeros(bias_vector.len()));
        }

        let mut nabla_w = Vec::new();
        for weight_matrix in &self.parameters.weights {
            nabla_w.push(DMatrix::<f64>::zeros(
                weight_matrix.nrows(),
                weight_matrix.ncols(),
            ));
        }

        let mut activation: DVector<f64> = x.clone(); //Startvektor
        let mut activations = vec![x.clone()]; //Liste mit allen Layern
        let mut zs = Vec::new(); //z Vektoren

        //Feedforward
        T::training_feedforward(&self.parameters, &mut activation, &mut activations, &mut zs);

        let mut delta = &activations.pop().unwrap() - y; //Cross-Entropy-Loss
        zs.pop();

        let nablas_size = nabla_w.len();
        nabla_b[nablas_size - 1] = delta.clone();
        nabla_w[nablas_size - 1] = &delta * &activations.pop().unwrap().transpose();

        for n in (0..=nablas_size - 2).rev() {
            let sp = T::get_derivative(&zs.pop().unwrap());
            delta = (&self.parameters.weights[n + 1].transpose() * &delta).component_mul(&sp);
            nabla_b[n] = delta.clone();
            nabla_w[n] = &delta * &activations.pop().unwrap().transpose();
        }
        (nabla_b, nabla_w)
    }

    pub fn update_parameters(
        &mut self,
        nabla_w: &[DMatrix<f64>],
        nabla_b: &[DVector<f64>],
        mini_batch_size: f64,
        trainig_dat_len: f64,
    ) {
        let mini_batch_learning_rate = self.learning_rate / mini_batch_size;

        for (w, nw) in zip(&mut self.parameters.weights, nabla_w) {
            R::reg_w(w, self.lambda, self.learning_rate, trainig_dat_len);
            *w -= mini_batch_learning_rate * nw; //Lerning Rate wird durch mini_batch Länge geteilt, weil nabla_w nur die Summe der 
            //Gradientengewichte hat. So wird der Mittelwert des Gradienten genommen
        }

        for (b, nb) in zip(&mut self.parameters.biases, nabla_b) {
            *b -= mini_batch_learning_rate * nb;
        }
    }

    pub fn get_weight_sizes(&self) -> &[(usize, usize)] {
        &self.weight_sizes
    }

    pub fn get_bias_sizes(&self) -> &[usize] {
        &self.bias_sizes
    }
}
