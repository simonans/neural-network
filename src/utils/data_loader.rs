#![allow(non_camel_case_types)]

use csv;
use nalgebra::{DMatrix, DVector, SimdValue};
use std::{
    error::Error,
    fs::File,
    io::{BufReader, Read},
};

use crate::cnn::Tensor;

pub enum MNIST_Data<'a> {
    Training(&'a mut Vec<(DVector<f64>, DVector<f64>)>),
    Test(&'a mut Vec<(DVector<f64>, usize)>),
}

pub enum Cifar_Data<'a> {
    Training(&'a mut Vec<(Tensor, DVector<f64>)>),
    Test(&'a mut Vec<(Tensor, usize)>),
}

pub fn get_mnist_data(path: &str, data: &mut MNIST_Data) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;

    for result in rdr.records() {
        let row = result?;
        match data {
            MNIST_Data::Training(d) => {
                let mut tmp = DVector::<f64>::zeros(10);
                tmp[row[0].parse::<usize>().unwrap()] = 1f64;

                d.push((
                    DVector::<f64>::from_vec(
                        row.iter()
                            .skip(1)
                            .map(|c| c.parse::<f64>().unwrap() / 127.5 - 1.0)
                            .collect(),
                    ),
                    tmp,
                ));
            }

            MNIST_Data::Test(d) => {
                d.push((
                    DVector::<f64>::from_vec(
                        row.iter()
                            .skip(1)
                            .map(|c| c.parse::<f64>().unwrap() / 127.5 - 1.0)
                            .collect(),
                    ),
                    row[0].parse().unwrap(),
                ));
            }
        }
    }

    Ok(())
}

pub fn get_cifar_data(paths: &Vec<&str>, data: &mut Cifar_Data) -> Result<(), Box<dyn Error>> {
    let mut buf = vec![0; 3073];
    for p in paths {
        let file = File::open(p)?;
        let mut reader = BufReader::new(file);
        loop {
            match reader.read_exact(&mut buf) {
                Ok(_) => extract_chunk(data, &buf),
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
        }
    }
    Ok(())
}

fn extract_chunk(data: &mut Cifar_Data, buf: &Vec<u8>) {
    match data {
        Cifar_Data::Training(d) => {
            let mut y = DVector::<f64>::zeros(10);
            y[buf[0] as usize] = 1f64;
            d.push((extract_colors(buf), y))
        }
        Cifar_Data::Test(d) => {
            let y = buf[0] as usize;
            d.push((extract_colors(buf), y));
        }
    }
}

fn extract_colors(buf: &Vec<u8>) -> Tensor {
    let converted: Vec<f64> = buf.iter().map(|x| *x as f64).collect();
    let mut colors = Vec::new();
    for n in 0..3 {
        let start = 1 + n * 1024;
        let channel = &converted[start..start + 1024];
        colors.push(DMatrix::from_row_slice(32, 32, channel));
    }
    Tensor::new(colors)
}
