use lazy_static::lazy_static;
use prometheus::Histogram;
use prometheus::{histogram_opts, register_histogram};

pub mod worker {
    use super::*;
    lazy_static! {
        pub static ref MODEL_STEP_DURATION: Histogram = register_histogram!(histogram_opts!(
            "worker_model_step_duration",
            "Model step duration distribution.",
            vec![40e-3, 50e-3, 60e-3, 75e-3, 80e-3, 0.1],
        ))
        .unwrap();
    }
}
