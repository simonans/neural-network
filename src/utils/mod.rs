mod data_loader;
mod math;

pub use data_loader::{Cifar_Data, MNIST_Data, get_cifar_data, get_mnist_data};
pub use math::{argmax, relu_matrix, relu_vec, sigmoid, softmax};
