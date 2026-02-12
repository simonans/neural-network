use nalgebra::DVector;

pub fn sigmoid(z: &DVector<f64>) -> DVector<f64> {
    let ret = z.map(|x| 1f64 / (1f64 + (-x).exp()));
    ret
}

pub fn softmax(vec: &DVector<f64>) -> DVector<f64>{
    let denominator = vec.iter().fold(0.0, |acc, x| acc + x.exp());
    DVector::from_iterator(vec.len(), vec.iter()
                                                    .map(|x| x.exp() / denominator))
}

pub fn relu(vec: &DVector<f64>) -> DVector<f64> {
    DVector::from_iterator(vec.len(), vec.iter().map(|x| x.max(0.0)))
}

pub fn argmax(vec: DVector<f64>) -> usize {
    let mut max = 0;
    for n in  0..vec.len() {
        if vec[n] > vec[max] {
            max = n;
        }
    }
    max
}