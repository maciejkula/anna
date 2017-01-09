//! C ABI for the anna library.

extern crate libc;

extern crate ndarray;
extern crate anna;

use libc::{c_float, uintptr_t};

use ndarray::{ArrayView2, ArrayView1};


unsafe fn create_array_view2<'a>(rows: uintptr_t,
                                 cols: uintptr_t,
                                 data: *const c_float)
                                 -> ArrayView2<'a, f32> {
    ArrayView2::from_shape_ptr((rows as usize, cols as usize), data as *const f32)
}


unsafe fn create_array_view1<'a>(cols: uintptr_t, data: *const c_float) -> ArrayView1<'a, f32> {
    ArrayView1::from_shape_ptr(cols, data as *const f32)
}


/// Create the random projection forest model.
///
/// The supplied data buffer must contain row-major entries
/// for a (rows x cols) matrix.
///
/// Returns a pointer to the fitted model.
/// It should be freed using the `free_model` function.
#[no_mangle]
pub extern "C" fn fit_model(max_leaf_size: uintptr_t,
                            num_trees: uintptr_t,
                            parallel: uintptr_t,
                            rows: uintptr_t,
                            cols: uintptr_t,
                            data_ptr: *mut c_float)
                            -> *const anna::RandomProjectionForest {

    let use_parallel = parallel != 0;
    let data = unsafe { create_array_view2(rows, cols, data_ptr) };

    let mut hyper = anna::Hyperparameters::new();
    hyper.max_leaf_size(max_leaf_size as usize)
        .num_trees(num_trees as usize);

    if use_parallel {
        hyper.parallel();
    }

    let model = Box::new(hyper.fit(data).expect("Unable to fit model."));

    Box::into_raw(model)
}

/// Run query.
///
/// The results are placed into the supplied `output` buffer. The returned
/// value indicates the number of results returned, which will be at most
/// the size of the supplied output buffer.
#[no_mangle]
pub extern "C" fn query_unsorted(model_ptr: *const anna::RandomProjectionForest,
                                 data_ptr: *const c_float,
                                 data_len: uintptr_t,
                                 output_ptr: *mut uintptr_t,
                                 output_len: uintptr_t)
                                 -> uintptr_t {

    let model = unsafe { &*model_ptr };
    let query = unsafe { create_array_view1(data_len, data_ptr) };
    let output = unsafe { std::slice::from_raw_parts_mut(output_ptr, output_len) };

    let results = model.query_unsorted(query).expect("Unable to query");

    for (&result, out) in results.iter().zip(output.iter_mut()) {
        *out = result;
    }

    std::cmp::min(results.len(), output.len()) as uintptr_t
}

/// Free the random projection forest model.
#[no_mangle]
pub extern "C" fn free_model(model: *mut anna::RandomProjectionForest) {
    let _ = unsafe { Box::from_raw(model) };
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
