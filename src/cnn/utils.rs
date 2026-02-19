use nalgebra::DMatrix;
use rand::seq::index;
use std::iter::zip;

pub struct Tensor {
    pub data: Vec<DMatrix<f64>>,
}

impl Tensor {
    pub fn new(data: Vec<DMatrix<f64>>) -> Self {
        Tensor { data }
    }

    pub fn learning(&mut self, learning_rate: f64, nabla: &Tensor) {
        for (rl, ll) in zip(&mut self.data, &nabla.data) {
            *rl -= learning_rate * ll;
        }
    }
}

pub fn tensor_convolution(tensor: &Tensor, filters: &Tensor) -> DMatrix<f64> {
    let (m_rows, m_cols) = tensor.data[0].shape();
    let (f_rows, f_cols) = filters.data[0].shape();
    let o_rows = m_rows - f_rows + 1;
    let o_cols = m_cols - f_cols + 1;

    let mut output = DMatrix::<f64>::zeros(o_rows, o_cols);
    for (matrix, filter) in zip(&tensor.data, &filters.data) {
        output += matrix_convolution(matrix, filter)
    }
    output
}

pub fn matrix_convolution(matrix: &DMatrix<f64>, filter: &DMatrix<f64>) -> DMatrix<f64> {
    let (m_rows, m_cols) = matrix.shape();
    let (f_rows, f_cols) = filter.shape();
    let o_rows = m_rows - f_rows + 1;
    let o_cols = m_cols - f_cols + 1;

    let mut output = DMatrix::<f64>::zeros(o_rows, o_cols);

    for row in 0..o_rows {
        for col in 0..o_cols {
            let mut sum = 0.0;
            for fr in 0..f_rows {
                for fc in 0..f_cols {
                    sum += matrix[(row + fr, col + fc)] * filter[(fr, fc)]
                }
            }
            output[(row, col)] = sum;
        }
    }
    output
}

pub fn max_pooling(input: DMatrix<f64>) -> DMatrix<f64> {
    let (rows, cols) = input.shape();
    assert!(rows % 2 == 0);
    assert!(cols % 2 == 0);
    let o_rows = rows / 2;
    let o_cols = cols / 2;
    let mut output = DMatrix::<f64>::zeros(o_rows, o_cols);

    for r in 0..o_rows {
        for c in 0..o_cols {
            let cur_r = 2 * r;
            let cur_c = 2 * c;
            output[(r, c)] = max(
                input[(cur_r, cur_c)],
                input[(cur_r + 1, cur_c)],
                input[(cur_r, cur_c + 1)],
                input[(cur_r + 1, cur_c + 1)],
            )
        }
    }
    output
}

fn max(v1: f64, v2: f64, v3: f64, v4: f64) -> f64 {
    v1.max(v2).max(v3).max(v4)
}

pub fn training_max_pooling(input: DMatrix<f64>) -> (DMatrix<f64>, Vec<(usize, usize)>) {
    //Pooling Matrx, (rows, cols)
    let (rows, cols) = input.shape();
    assert!(rows % 2 == 0);
    assert!(cols % 2 == 0);
    let o_rows = rows / 2;
    let o_cols = cols / 2;

    let mut output = DMatrix::<f64>::zeros(o_rows, o_cols);
    let mut inidzes = Vec::new();

    for r in 0..o_rows {
        for c in 0..o_cols {
            let cur_r = 2 * r;
            let cur_c = 2 * c;
            let m = training_max(&input, cur_r, cur_c);
            output[(r, c)] = m.0;
            inidzes.push(m.1);
        }
    }
    (output, inidzes)
}

fn training_max(input: &DMatrix<f64>, cur_r: usize, cur_c: usize) -> (f64, (usize, usize)) {
    let candidates = [
        (cur_r, cur_c),
        (cur_r + 1, cur_c),
        (cur_r, cur_c + 1),
        (cur_r + 1, cur_c + 1),
    ];

    let idx = candidates
        .into_iter()
        .max_by(|a, b| input[*a].total_cmp(&input[*b]))
        .unwrap();
    (input[idx], idx)
}

pub fn relu_derivative(input: &Tensor) -> Tensor {
    let mut data = Vec::new();
    for m in &input.data {
        data.push(DMatrix::from_iterator(
            m.nrows(),
            m.ncols(),
            m.iter().map(|x| if *x < 0.0 { 0.0 } else { 1.0 }),
        ));
    }
    Tensor { data }
}
