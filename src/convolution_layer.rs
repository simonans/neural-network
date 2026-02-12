use nalgebra::DMatrix;

pub struct FeatureMap {
    data: Vec<DMatrix<f64>>
}

impl FeatureMap {
    pub fn new(data: Vec<DMatrix<f64>>) -> Self {
        FeatureMap { data }
    }
}