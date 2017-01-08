mod errors;
mod trees;

extern crate rand;

#[macro_use(s)]
extern crate ndarray;
extern crate ndarray_rand;

use ndarray::{Array, Array1, Array2, ArrayView2, ArrayView1};

use rand::Rng;
use rand::distributions::{IndependentSample, Normal};

pub use errors::{Error, ErrorType, Result};
use trees::Tree;


pub struct Hyperparameters {
    max_leaf_size: usize,
    num_trees: usize,
}


impl Hyperparameters {
    pub fn new() -> Hyperparameters {
        Hyperparameters {
            max_leaf_size: 10,
            num_trees: 10,
        }
    }

    pub fn max_leaf_size(&mut self, max_leaf_size: usize) -> &mut Hyperparameters {
        self.max_leaf_size = max_leaf_size;
        self
    }

    pub fn num_trees(&mut self, num_trees: usize) -> &mut Hyperparameters {
        self.num_trees = num_trees;
        self
    }

    pub fn fit(&self, data: ArrayView2<f32>) -> Result<Raproxy> {

        if !Self::has_finite_entries(data) {
            return Err(Error { error_type: ErrorType::NonFiniteEntry });
        }

        if !Self::has_valid_norms(data) {
            return Err(Error { error_type: ErrorType::ZeroNorm });
        }

        let raproxy = Raproxy {
            max_leaf_size: self.max_leaf_size,
            num_trees: self.num_trees,
            dim: data.cols(),
            trees: Vec::new()
        };

        let mut trees = Vec::with_capacity(self.num_trees);

        for _ in 0..self.num_trees {
            trees.push(Tree::new(self.max_leaf_size,
                                 data));
        }

        Ok(raproxy)
    }

    fn has_finite_entries(data: ArrayView2<f32>) -> bool {
        for elem in data.iter() {
            if !elem.is_finite() {
                return false
            }
        }

        true
    }

    fn has_valid_norms(data: ArrayView2<f32>) -> bool {
        for row in data.inner_iter() {
            if !row.dot(&row).is_normal() {
                return false
            }
        }

        true
    }
}

pub struct Raproxy {
    max_leaf_size: usize,
    num_trees: usize,
    dim: usize,
    trees: Vec<Tree>,
}


#[cfg(test)]
mod tests {

    use rand::distributions::normal::Normal;

    use ndarray::Array;
    use ndarray_rand::{RandomExt, F32};

    use super::*;

    #[test]
    fn it_works() {

        let arr = Array::zeros((5, 2));
        let first_row = arr.row(0);

        let products = arr.inner_iter().map(|row| row.dot(&first_row)).collect::<Vec<f32>>();

        println!("{:#?}", products);
    }

    #[test]
    fn generate_input() {

        let data = Array::random((100, 10), F32(Normal::new(0.0, 1.0)));
        let proxy = Hyperparameters::new().fit(data.view());
    }

    #[test]
    fn no_overflow_on_bad_splits() {
        let data = Array::zeros((100, 10)) + 1.0;
        let proxy = Hyperparameters::new().fit(data.view());
    }

    #[test]
    fn throw_error_on_zero_vectors() {
        let data = Array::zeros((100, 10));
        match Hyperparameters::new().fit(data.view()) {
            Ok(model) => panic!("Should have errored out."),
            Err(error) => assert!(error.error_type == ErrorType::ZeroNorm),
        }
    }
}
