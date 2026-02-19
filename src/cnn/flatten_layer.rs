use std::process::id;

use nalgebra::{DMatrix, DVector};

use super::utils::Tensor;

pub struct FlattenLayer {
    w: usize,
    h: usize,
    d: usize,
}

impl FlattenLayer {
    pub fn new(w: usize, h: usize, d: usize) -> Self {
        Self { w, h, d }
    }

    pub fn flatten(&self, tensor: Tensor) -> DVector<f64> {
        let mut data = Vec::new();
        for m in tensor.data {
            for e in &m {
                data.push(*e);
            }
        }
        DVector::from_vec(data)
    }

    pub fn back(&self, vector: DVector<f64>) -> Tensor {
        let mut data = vec![DMatrix::zeros(self.h, self.w); self.d];
        let mut idx = 0;
        for d in 0..self.d {
            let matrix = &mut data[d];
            for col in 0..self.w {
                for row in 0..self.h {
                    matrix[(row, col)] = vector[idx];
                    idx += 1;
                }
            }
        }

        Tensor { data }
    }
}
