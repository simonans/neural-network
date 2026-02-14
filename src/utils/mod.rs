mod data_loader;
mod math;

pub use data_loader::{MNIST_Data, get_mnist_data};
pub use math::{argmax, relu, sigmoid, softmax};
