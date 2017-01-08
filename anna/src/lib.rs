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

    pub fn fit(&self, data: ArrayView2<f32>) -> Result<RandomProjectionForest> {

        if !Self::has_finite_entries(data) {
            return Err(Error { error_type: ErrorType::NonFiniteEntry });
        }

        if !Self::has_valid_norms(data) {
            return Err(Error { error_type: ErrorType::ZeroNorm });
        }

        let mut raproxy = RandomProjectionForest {
            max_leaf_size: self.max_leaf_size,
            num_trees: self.num_trees,
            dim: data.cols(),
            trees: Vec::new(),
        };

        let mut trees = Vec::with_capacity(self.num_trees);

        for _ in 0..self.num_trees {
            trees.push(Tree::new(self.max_leaf_size, data));
        }

        raproxy.trees = trees;

        Ok(raproxy)
    }

    fn has_finite_entries(data: ArrayView2<f32>) -> bool {
        for elem in data.iter() {
            if !elem.is_finite() {
                return false;
            }
        }

        true
    }

    fn has_valid_norms(data: ArrayView2<f32>) -> bool {
        for row in data.inner_iter() {
            if !row.dot(&row).is_normal() {
                return false;
            }
        }

        true
    }
}

pub struct RandomProjectionForest {
    max_leaf_size: usize,
    num_trees: usize,
    dim: usize,
    trees: Vec<Tree>,
}


impl RandomProjectionForest {
    pub fn query(&self, query_vector: ArrayView1<f32>) -> Vec<usize> {

        let query_norm = query_vector.dot(&query_vector).sqrt();
        let mut merged_vector = Vec::with_capacity(self.max_leaf_size * self.num_trees);

        for subresult in self.trees
            .iter()
            .map(|tree| tree.query(query_vector, query_norm)) {
            merged_vector.extend_from_slice(subresult);
        }

        merged_vector.sort();
        merged_vector.dedup();

        merged_vector
    }
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
        let model = Hyperparameters::new().fit(data.view());
    }

    #[test]
    fn no_overflow_on_bad_splits() {
        let data = Array::zeros((100, 10)) + 1.0;
        let model = Hyperparameters::new().fit(data.view());
    }

    #[test]
    fn throw_error_on_zero_vectors() {
        let data = Array::zeros((100, 10));
        match Hyperparameters::new().fit(data.view()) {
            Ok(model) => panic!("Should have errored out."),
            Err(error) => assert!(error.error_type == ErrorType::ZeroNorm),
        }
    }

    #[test]
    fn self_returned_from_query() {

        let data = Array::random((1000, 10), F32(Normal::new(0.0, 1.0)));
        let model = Hyperparameters::new().fit(data.view()).unwrap();

        for idx in 0..data.rows() {
            assert!(model.query(data.row(idx)).contains(&idx))
        }
    }
}
