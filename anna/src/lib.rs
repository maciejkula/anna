mod errors;
mod trees;

extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate rayon;

use ndarray::{ArrayView2, ArrayView1};

use rayon::prelude::*;

pub use errors::{Error, ErrorType, Result};
use trees::Tree;


pub struct Hyperparameters {
    max_leaf_size: usize,
    num_trees: usize,
    parallel: bool,
}


impl Hyperparameters {
    pub fn new() -> Hyperparameters {
        Hyperparameters {
            max_leaf_size: 10,
            num_trees: 10,
            parallel: false,
        }
    }

    pub fn max_leaf_size(&mut self, max_leaf_size: usize) -> &mut Hyperparameters {
        assert!(max_leaf_size > 0,
                "Max leaf size must be greater than zero.");
        self.max_leaf_size = max_leaf_size;
        self
    }

    pub fn num_trees(&mut self, num_trees: usize) -> &mut Hyperparameters {
        assert!(num_trees > 0, "Number of trees must be greater than zero.");
        self.num_trees = num_trees;
        self
    }

    pub fn parallel(&mut self) -> &mut Hyperparameters {
        self.parallel = true;
        self
    }

    pub fn fit(&self, data: ArrayView2<f32>) -> Result<RandomProjectionForest> {

        if !Self::has_finite_entries(data) {
            return Err(Error::new(ErrorType::NonFiniteEntry));
        }

        if !Self::has_valid_norms(data) {
            return Err(Error::new(ErrorType::ZeroNorm));
        }

        let trees = if self.parallel {
            (0..self.num_trees)
                .into_par_iter()
                .map(|_| Tree::new(self.max_leaf_size, data))
                .collect::<Vec<_>>()
        } else {
            (0..self.num_trees)
                .map(|_| Tree::new(self.max_leaf_size, data))
                .collect::<Vec<_>>()
        };

        let model = RandomProjectionForest {
            max_leaf_size: self.max_leaf_size,
            num_trees: self.num_trees,
            dim: data.cols(),
            trees: trees,
        };

        Ok(model)
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
    pub fn query(&self, query_vector: ArrayView1<f32>) -> Result<Vec<usize>> {

        if query_vector.len() != self.dim {
            return Err(Error::new(ErrorType::IncompatibleDimensions(format!("Incompatible \
                                                                             dimensions: \
                                                                             expected {} but \
                                                                             got {}",
                                                                            self.dim,
                                                                            query_vector.len()))));
        }

        let query_norm = query_vector.dot(&query_vector).sqrt();
        let mut merged_vector = Vec::with_capacity(self.max_leaf_size * self.num_trees);

        for subresult in self.trees
            .iter()
            .map(|tree| tree.query(query_vector, query_norm)) {
            merged_vector.extend_from_slice(subresult);
        }

        merged_vector.sort();
        merged_vector.dedup();

        Ok(merged_vector)
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
        let _ = Hyperparameters::new().fit(data.view());
    }

    #[test]
    fn no_overflow_on_bad_splits() {
        let data = Array::zeros((100, 10)) + 1.0;
        let _ = Hyperparameters::new().fit(data.view());
    }

    #[test]
    fn throw_error_on_zero_vectors() {
        let data = Array::zeros((100, 10));
        match Hyperparameters::new().fit(data.view()) {
            Ok(_) => panic!("Should have errored out."),
            Err(error) => assert!(error.error_type == ErrorType::ZeroNorm),
        }
    }

    #[test]
    fn self_returned_from_query() {

        let max_leaf_size = 10;
        let num_trees = 5;

        let data = Array::random((3000, 10), F32(Normal::new(0.0, 1.0)));

        let model = Hyperparameters::new()
            .max_leaf_size(max_leaf_size)
            .num_trees(num_trees)
            .parallel()
            .fit(data.view())
            .unwrap();

        for idx in 0..data.rows() {
            let results = model.query(data.row(idx)).unwrap();
            assert!(results.len() < max_leaf_size * num_trees);
            assert!(results.contains(&idx));
        }
    }
}
