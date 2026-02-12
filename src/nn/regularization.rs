#![allow(dead_code)]

use nalgebra::DMatrix;
use num_traits::signum;

pub trait Regularization {
    fn reg_w(w: &mut DMatrix<f64>, lambda: &f64, eta: &f64, n: &f64);
}

pub struct NON {}
impl Regularization for NON {
    fn reg_w(_: &mut DMatrix<f64>, _: &f64, _: &f64, _: &f64) {
      
    }
}
pub struct L1 {}
impl Regularization for L1 {
    fn reg_w(w: &mut DMatrix<f64>, lambda: &f64, eta: &f64, n: &f64) {
        let factor = - eta * (lambda/n);
        w.apply(|w| *w -= factor * signum(*w));
    }
}

pub struct L2 {}
impl Regularization for L2 {
    fn reg_w(w: &mut DMatrix<f64>, lambda: &f64, eta: &f64, n: &f64) {
        *w *= 1.0 - eta * (lambda/n);
    }
}