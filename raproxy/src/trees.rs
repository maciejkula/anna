use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Normal};

use ndarray::{Array, Array1, Array2, ArrayView2, ArrayView1};


#[derive(Debug)]
struct Split {
    plane_idx: usize,
    intercept: f32,
    children: (Box<Node>, Box<Node>),
}

#[derive(Debug)]
struct Leaf {
    point_ids: Vec<usize>,
}

#[derive(Debug)]
enum Node {
    Split(Split),
    Leaf(Leaf),
    Empty
}

struct Hyperplanes {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}


impl Hyperplanes {
    fn new(cols: usize) -> Hyperplanes {
        Hyperplanes {
            rows: 0,
            cols: cols,
            data: Vec::new(),
        }
    }

    fn has_hyperplane(&self, idx: usize) -> bool {
        idx < self.rows
    }

    fn get(&self, idx: usize) -> Option<ArrayView1<f32>> {

        if idx < self.rows {
            let start = idx * self.cols;
            let stop = start + self.cols;

            Some(ArrayView1::from_shape(self.cols, &self.data[start..stop])
                .expect("Incompatible shape when getting hyperplane"))
        } else {
            None
        }
    }

    fn add<R: Rng>(&mut self, random_state: &mut R) -> ArrayView1<f32> {
        let normal = Normal::new(0.0, 1.0);

        let hyperplanes_len = self.data.len();
        let mut norm = 0.0;

        self.data.reserve(self.cols);

        for _ in 0..self.cols {
            let entry = normal.ind_sample(random_state) as f32;

            self.data.push(entry);

            norm += entry.powi(2);
        }

        norm = norm.sqrt();

        // Normalize to unit length
        for entry in &mut self.data[hyperplanes_len..] {
            *entry /= norm
        }

        self.rows += 1;

        self.get(self.rows - 1).expect("Failed to add new hyperplane")
    }
}

pub struct Tree {
    max_leaf_size: usize,
    hyperplanes: Hyperplanes,
    root: Node
}


impl Tree {
    pub fn new(max_leaf_size: usize, data: ArrayView2<f32>) -> Tree {
        let mut tree = Tree {
            max_leaf_size: max_leaf_size,
            hyperplanes: Hyperplanes::new(data.cols()),
            root: Node::Empty
        };

        let indices = (0..data.rows()).collect::<Vec<usize>>();
        let norms = data.inner_iter().map(|row| row.dot(&row).sqrt()).collect::<Vec<f32>>();

        tree.root = tree.fit_tree(indices,
                                  data,
                                  &norms,
                                  0);
        
        println!("{:#?}", tree.root);
        tree
    }

    fn fit_tree(&mut self,
                indices: Vec<usize>,
                data: ArrayView2<f32>,
                norms: &[f32],
                depth: usize) -> Node {

        println!("fitting at depth {}", depth);

        if indices.len() <= self.max_leaf_size {
            return Node::Leaf(Leaf { point_ids: indices })
        }
        
        let mut random_state = rand::thread_rng();

        let point_distances = {
            let hyperplane = if self.hyperplanes.has_hyperplane(depth) {
                self.hyperplanes.get(depth).expect("Failed to get hyperplane with a valid index.")
            } else {
                self.hyperplanes.add(&mut random_state)
            };

            let mut distances = indices.iter()
                .map(|&idx| (idx, data.row(idx).dot(&hyperplane) / norms[idx]))
                .collect::<Vec<_>>();

            distances.sort_by(|a, b| {
                a.partial_cmp(b).expect("NaN values encountered when creating tree splits")
            });

            distances
        };

        if !Self::valid_split(&point_distances) {
            Node::Leaf(Leaf { point_ids: indices })
        } else {
            let (_, intercept) = point_distances[point_distances.len() / 2];
            let mut left_indices = Vec::new();
            let mut right_indices = Vec::new();

            for &(idx, distance) in &point_distances {
                if distance <= intercept {
                    left_indices.push(idx);
                } else {
                    right_indices.push(idx);
                }
            }

            Node::Split(Split {
                plane_idx: depth,
                intercept: intercept,
                children: (Box::new(self.fit_tree(left_indices, data, norms, depth + 1)),
                           Box::new(self.fit_tree(right_indices, data, norms, depth + 1)))
            })
        }
    }

    fn valid_split(distances: &[(usize, f32)]) -> bool {
        match (distances.first(), distances.last()) {
            (Some(&(_, smallest)), Some(&(_, largest))) => smallest != largest,
            _ => false,
        }
    }
}
