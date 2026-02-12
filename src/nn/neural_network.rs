use nalgebra::{DMatrix, DVector};
use rand::{rng, seq::SliceRandom, Rng};
use std::{iter::zip, marker::PhantomData};

use super::network_architecture::{Parameters, NetworkArchitecture};
use super::regularization::{Regularization};

use crate::utils::argmax;

pub struct NeuralNetwork<T: NetworkArchitecture, R: Regularization> {
    parameters: Parameters,
    architecture: PhantomData<T>,
    regularization: PhantomData<R>
}

impl<T: NetworkArchitecture, R: Regularization> NeuralNetwork<T, R> {
    pub fn new(sizes: Vec<usize>) -> Self {
        let mut biases = Vec::new();
        let mut weights = Vec::new();
        let mut rng = rand::rng();
        for n in 0..(sizes.len() - 1) {
            weights.push(DMatrix::<f64>::from_fn(
                sizes[n+1], 
                sizes[n],
                |_,_| {
                    let scale = (1.0 / sizes[n] as f64).sqrt();
                    rng.random_range(-scale..scale) //Gauß
                }));
            //w11 w21 w31 
            //w12 w22 w32  Höhe = Rows = Anzahl Neuronen rechter Layer (Output)
            //w13 w23 w33  Breite = Cols = Anzahl Neuronen linker Layer (Input)
            //w14 w24 w34  Hier mapt 3 Neuronen auf 4 Neuronen

            biases.push(DVector::<f64>::zeros(
                sizes[n+1]));
            //Input Layer hat keine Gewichte
        }

        NeuralNetwork { 
            parameters: Parameters::new(weights, biases),
            architecture: PhantomData,
            regularization: PhantomData
        }
    }

    pub fn stochastic_gradient_descent(&mut self, 
                mut training_data: Vec<(DVector<f64>, DVector<f64>)>, //Input data vector with corresponding desired output vector
                epochs: i32,
                mini_batch_size: usize,
                eta: f64,
                lambda: f64
            ) {
        for e in 0..epochs {
            training_data.shuffle(&mut rng());
            for mini_batch in training_data.chunks(mini_batch_size) {
                self.update_mini_batch(
                                    mini_batch, 
                                    eta, 
                                    lambda,
                                    training_data.len() as f64);
            }
            println!("SGD: Epoch {} done", e);
        }

    }

    fn update_mini_batch(&mut self, 
                mini_batch: &[(DVector<f64>, DVector<f64>)],
                eta: f64,
                lambda: f64,
                n: f64
            ) {
        let mut nabla_b = Vec::new();
        for bias_vector in &self.parameters.biases {
            nabla_b.push(DVector::<f64>::zeros(bias_vector.len()));
        }

        let mut nabla_w = Vec::new();
        for weight_matrix in &self.parameters.weights {
            nabla_w.push(DMatrix::<f64>::zeros(weight_matrix.nrows(), weight_matrix.ncols()));
        }

        //Alle Datensätze im mini_batch
        for (x,y) in mini_batch {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(x, y);
            for (nb, dnb) in zip(&mut nabla_b, &delta_nabla_b) {
                *nb += dnb;
            }
            for (nw, dnw) in zip(&mut nabla_w, delta_nabla_w) {
                *nw += dnw;
            }
        }

        let mini_batch_learning_rate = eta / mini_batch.len() as f64;

        for (w, nw) in zip(&mut self.parameters.weights, &nabla_w) {
            R::reg_w(w, &lambda, &eta, &n);
            *w -= mini_batch_learning_rate * nw; //Lerning Rate wird durch mini_batch Länge geteilt, weil nabla_w nur die Summe der 
                                                        //Gradientengewichte hat. So wird der Mittelwert des Gradienten genommen
        }

        for (b, nb) in zip(&mut self.parameters.biases, &nabla_b) {
            *b -= mini_batch_learning_rate * nb;
        }

    }

    fn backprop(&self, x: &DVector<f64>, y: &DVector<f64>) ->(Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
       let mut nabla_b = Vec::new();
        for bias_vector in &self.parameters.biases {
            nabla_b.push(DVector::<f64>::zeros(bias_vector.len()));
        }

        let mut nabla_w = Vec::new();
        for weight_matrix in &self.parameters.weights {
            nabla_w.push(DMatrix::<f64>::zeros(weight_matrix.nrows(), weight_matrix.ncols()));
        } 

        let mut activation: DVector<f64> = x.clone(); //Startvektor
        let mut activations = vec![x.clone()];  //Liste mit allen Layern
        let mut zs = Vec::new();    //z Vektoren

        //Feedforward
        T::training_feedforward(&self.parameters, &mut activation, &mut activations, &mut zs);

        let mut delta = &activations.pop().unwrap() - y;    //Cross-Entropy-Loss
        zs.pop(); 

        let nablas_size = nabla_w.len(); 
        nabla_b[nablas_size - 1] = delta.clone();
        nabla_w[nablas_size - 1] = &delta * &activations.pop().unwrap().transpose();
        
         
        for n in (0..=nablas_size-2).rev() {    
            let sp = T::get_derivative(&zs.pop().unwrap());
            delta = (&self.parameters.weights[n+1].transpose() * &delta).component_mul(&sp);
            nabla_b[n] = delta.clone();
            nabla_w[n] = &delta * &activations.pop().unwrap().transpose();
        }
        (nabla_b, nabla_w)
    }

    pub fn evaluate(&self, test_data: Vec<(DVector<f64>, usize)>) -> usize {
        let mut erg: Vec<(usize, usize)> = Vec::new();
        for (x,y) in test_data {
            erg.push((argmax(T::feedforward(&self.parameters, x)), y));
        }
        let mut sum: usize = 0;
        for (x,y) in &erg {
            if x == y {
                sum += 1;
            }
        }
        sum
    }

}

