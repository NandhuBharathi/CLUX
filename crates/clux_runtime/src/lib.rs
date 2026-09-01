pub mod telemetry;
pub mod layer;
pub mod engine;

use pyo3::prelude::*;
use engine::SsmTrainingEngine;

#[pyclass]
pub struct SovereignModel {
    engine: SsmTrainingEngine,
}

#[pymethods]
impl SovereignModel {
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        let engine = SsmTrainingEngine::load_from_file(model_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(SovereignModel { engine })
    }

    fn generate(&mut self, prompt: &str, count: usize, temp: f32, top_k: usize) -> PyResult<String> {
        let result = self.engine.generate(prompt, count, temp, top_k);
        Ok(result)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn clux(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SovereignModel>()?;
    Ok(())
}
