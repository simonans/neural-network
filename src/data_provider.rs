use std::error::Error;
use nalgebra::DVector;
use csv;

pub enum Data<'a> {
    Training(&'a mut Vec<(DVector<f64>, DVector<f64>)>),
    Test(&'a mut Vec<(DVector<f64>, usize)>)
}

pub fn get_data(path: &str, data: & mut Data) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;

    for result in rdr.records() {
        let row = result?;
        match data {
            Data::Training(d) => {
                let mut tmp = DVector::<f64>::zeros(10);
                tmp[row[0].parse::<usize>().unwrap()] = 1f64;

                d.push((DVector::<f64>::from_vec(row
                                .iter()
                                .skip(1)
                                .map(|c| c.parse::<f64>().unwrap() / 127.5 - 1.0 )
                                .collect()), 
                        tmp));
               
            }

            Data::Test(d) => {
                d.push((DVector::<f64>::from_vec(row
                                .iter()
                                .skip(1)
                                .map(|c| c.parse::<f64>().unwrap() / 127.5 - 1.0 )
                                .collect()), 
                row[0].parse().unwrap()));
            }
        }
    }

    Ok(())
}