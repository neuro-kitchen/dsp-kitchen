//! `OnnxGraphRunner`: Zero-copy `.onnx` graph parser (`onnx-ir = "0.21.0"`) and
//! dynamic Burn tensor execution engine for running external spike-sorting models
//! (Kilosort4, DARTsort, CEBRA, Bombcell/UnitMatch) inside `dsp-synapse-ml`.

use std::collections::HashMap;
use std::path::Path;
use anyhow::{Context, Result, bail};
use onnx_ir::ir::{ArgType, Argument, Node, OnnxGraph};
use onnx_ir::OnnxGraphBuilder;
use serde::{Deserialize, Serialize};
use crate::backend::{SynapseMlDevice, Tensor2D, Tensor3D};

mod ops;
#[path = "proto_builder.rs"]
pub mod onnx_proto_builder;
mod tensor;

pub use tensor::DynTensor;
use ops::conv::{exec_avg_pool1d, exec_conv1d};
use ops::elementwise::broadcast_binary_op;
use ops::linear::{exec_gemm, exec_linear};
use ops::norm::{exec_batch_norm, exec_layer_norm, exec_softmax};
use ops::shape::{exec_concat, exec_transpose, resolve_reshape_dims};

/// Metadata describing a graph input or output port in an [`OnnxGraphRunner`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnnxPortSpec {
    pub name: String,
    pub rank: usize,
    pub static_shape: Option<Vec<Option<usize>>>,
}


/// Burn-ONNX (`onnx-ir = "0.21.0"`) runtime graph executor.
///
/// Parses standard `.onnx` files (via `memmap2` zero-copy loading) or in-memory `.onnx`
/// buffers into a simplified `onnx_ir::OnnxGraph` and executes its topologically sorted
/// operator nodes on [`Tensor`] / `burn_tensor`.
#[derive(Debug, Clone)]
pub struct OnnxGraphRunner {
    graph: OnnxGraph,
    pub device: SynapseMlDevice,
    inputs: Vec<OnnxPortSpec>,
    outputs: Vec<OnnxPortSpec>,
    /// Initializers and constants decoded once at load, by argument name.
    constants: HashMap<String, DynTensor>,
    i64_constants: HashMap<String, Vec<i64>>,
}

impl OnnxGraphRunner {
    /// Parses a `.onnx` file from disk using `onnx_ir::OnnxGraphBuilder` (with `memmap2`
    /// zero-copy loading and ONNX graph simplification passes enabled).
    pub fn from_file(path: impl AsRef<Path>, device: SynapseMlDevice) -> Result<Self> {
        let path_ref = path.as_ref();
        let graph = OnnxGraphBuilder::new()
            .simplify(true)
            .parse_file(path_ref)
            .with_context(|| format!("Failed to parse ONNX file '{}'", path_ref.display()))?;
        Ok(Self::from_graph(graph, device))
    }

    /// Parses an in-memory `.onnx` protobuf byte slice using `onnx_ir::OnnxGraphBuilder`.
    pub fn from_bytes(bytes: &[u8], device: SynapseMlDevice) -> Result<Self> {
        let graph = OnnxGraphBuilder::new()
            .simplify(true)
            .parse_bytes(bytes)
            .context("Failed to parse ONNX model bytes")?;
        Ok(Self::from_graph(graph, device))
    }

    /// Wraps a pre-parsed [`onnx_ir::OnnxGraph`], decoding its initializers and constants once.
    pub fn from_graph(graph: OnnxGraph, device: SynapseMlDevice) -> Self {
        let inputs = graph.inputs.iter().map(port_spec_from_arg).collect();
        let outputs = graph.outputs.iter().map(port_spec_from_arg).collect();
        let mut constants = HashMap::new();
        let mut i64_constants = HashMap::new();
        for node in &graph.nodes {
            for arg in node.inputs().iter().chain(node.outputs()) {
                if arg.name.is_empty() || constants.contains_key(&arg.name) || i64_constants.contains_key(&arg.name) {
                    continue;
                }
                match decode_constant(arg, device) {
                    Some(Constant::F32(t)) => {
                        constants.insert(arg.name.clone(), t);
                    }
                    Some(Constant::I64(v)) => {
                        i64_constants.insert(arg.name.clone(), v);
                    }
                    None => {}
                }
            }
        }
        Self { graph, device, inputs, outputs, constants, i64_constants }
    }

    pub fn graph(&self) -> &OnnxGraph {
        &self.graph
    }

    pub fn input_specs(&self) -> &[OnnxPortSpec] {
        &self.inputs
    }

    pub fn output_specs(&self) -> &[OnnxPortSpec] {
        &self.outputs
    }

    pub fn node_count(&self) -> usize {
        self.graph.nodes.len()
    }

    /// Convenience wrapper for single-input, single-output 2D `[N, F_in] -> [N, F_out]` models.
    pub fn run_2d(&self, input: &Tensor2D) -> Result<Tensor2D> {
        let out = self.run_single(DynTensor::from_tensor(input))?;
        out.into_tensor::<2>()
    }

    /// Convenience wrapper for single-input, single-output 3D `[N, C_in, T_in] -> [N, C_out, T_out]` models.
    pub fn run_3d(&self, input: &Tensor3D) -> Result<Tensor3D> {
        let out = self.run_single(DynTensor::from_tensor(input))?;
        out.into_tensor::<3>()
    }

    /// Convenience wrapper for a 3D input `[N, K, T]` producing a 2D output `[N, D]` (e.g., embedders, detectors, localizers).
    pub fn run_3d_to_2d(&self, input: &Tensor3D) -> Result<Tensor2D> {
        let out = self.run_single(DynTensor::from_tensor(input))?;
        out.into_tensor::<2>()
    }

    /// Executes a single-input, single-output ONNX graph.
    pub fn run_single(&self, input: DynTensor) -> Result<DynTensor> {
        let in_name = self
            .inputs
            .first()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "input".to_string());
        let mut feed = HashMap::with_capacity(1);
        feed.insert(in_name, input);
        let mut outputs = self.run(feed)?;
        let out_name = self
            .outputs
            .first()
            .map(|p| p.name.as_str())
            .unwrap_or("output");
        outputs
            .remove(out_name)
            .or_else(|| outputs.into_values().next())
            .context("ONNX graph produced no outputs")
    }

    /// Executes the ONNX graph given named input tensors and returns all graph outputs by name.
    pub fn run(&self, mut values: HashMap<String, DynTensor>) -> Result<HashMap<String, DynTensor>> {
        let mut i64_values: HashMap<String, Vec<i64>> = HashMap::new();

        for node in &self.graph.nodes {
            self.execute_node(node, &mut values, &mut i64_values)
                .with_context(|| format!("Failed executing ONNX node '{}'", node.name()))?;
        }

        let mut out_map = HashMap::with_capacity(self.graph.outputs.len());
        for out_arg in &self.graph.outputs {
            if let Some(val) = values.remove(&out_arg.name) {
                out_map.insert(out_arg.name.clone(), val);
            } else if let Ok(static_val) = self.resolve_f32_arg(out_arg, &values) {
                out_map.insert(out_arg.name.clone(), static_val);
            } else {
                bail!(
                    "Graph output '{}' was not produced during execution",
                    out_arg.name
                );
            }
        }
        Ok(out_map)
    }

    /// A runtime value or constant (shared, no copy); unnamed constants are decoded on demand.
    fn resolve_f32_arg(&self, arg: &Argument, values: &HashMap<String, DynTensor>) -> Result<DynTensor> {
        if !arg.name.is_empty()
            && let Some(v) = values.get(&arg.name).or_else(|| self.constants.get(&arg.name))
        {
            return Ok(v.clone());
        }
        match decode_constant(arg, self.device) {
            Some(Constant::F32(t)) => Ok(t),
            _ => bail!("Missing runtime or constant f32 tensor for argument '{}'", arg.name),
        }
    }

    fn resolve_i64_arg(&self, arg: &Argument, i64_values: &HashMap<String, Vec<i64>>) -> Result<Vec<i64>> {
        if !arg.name.is_empty()
            && let Some(v) = i64_values.get(&arg.name).or_else(|| self.i64_constants.get(&arg.name))
        {
            return Ok(v.clone());
        }
        match decode_constant(arg, self.device) {
            Some(Constant::I64(v)) => Ok(v),
            _ => bail!("Missing i64 tensor/shape for argument '{}'", arg.name),
        }
    }

    /// Elementwise operator: `output = f(input)`.
    fn unary(
        &self,
        input: &Argument,
        output: &Argument,
        values: &mut HashMap<String, DynTensor>,
        f: impl Fn(f32) -> f32,
    ) -> Result<()> {
        let x = self.resolve_f32_arg(input, values)?;
        values.insert(output.name.clone(), x.map(f));
        Ok(())
    }

    fn execute_node(
        &self,
        node: &Node,
        values: &mut HashMap<String, DynTensor>,
        i64_values: &mut HashMap<String, Vec<i64>>,
    ) -> Result<()> {
        match node {
            Node::Constant(n) => {
                let out_arg = &n.outputs[0];
                let src_arg = n.inputs.first().unwrap_or(out_arg);
                if let Ok(t) = self.resolve_f32_arg(out_arg, values).or_else(|_| self.resolve_f32_arg(src_arg, values)) {
                    values.insert(out_arg.name.clone(), t);
                } else if let Ok(v) = self.resolve_i64_arg(out_arg, i64_values).or_else(|_| self.resolve_i64_arg(src_arg, i64_values)) {
                    i64_values.insert(out_arg.name.clone(), v);
                }
            }
            Node::Identity(n) => {
                if let Ok(x) = self.resolve_f32_arg(&n.inputs[0], values) {
                    values.insert(n.outputs[0].name.clone(), x);
                } else if let Ok(ix) = self.resolve_i64_arg(&n.inputs[0], i64_values) {
                    i64_values.insert(n.outputs[0].name.clone(), ix);
                }
            }
            Node::Dropout(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                values.insert(n.outputs[0].name.clone(), x);
            }
            Node::Relu(n) => self.unary(&n.inputs[0], &n.outputs[0], values, |v| v.max(0.0))?,
            Node::LeakyRelu(n) => {
                let alpha = n.config.alpha as f32;
                self.unary(&n.inputs[0], &n.outputs[0], values, |v| if v >= 0.0 { v } else { alpha * v })?
            }
            Node::Gelu(n) => {
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                self.unary(&n.inputs[0], &n.outputs[0], values, |v| {
                    0.5 * v * (1.0 + (SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)).tanh())
                })?
            }
            Node::Sigmoid(n) => self.unary(&n.inputs[0], &n.outputs[0], values, |v| 1.0 / (1.0 + (-v).exp()))?,
            Node::Tanh(n) => self.unary(&n.inputs[0], &n.outputs[0], values, f32::tanh)?,
            Node::Swish(n) => self.unary(&n.inputs[0], &n.outputs[0], values, |v| v / (1.0 + (-v).exp()))?,
            Node::Elu(n) => {
                let alpha = n.config.alpha as f32;
                self.unary(&n.inputs[0], &n.outputs[0], values, |v| if v >= 0.0 { v } else { alpha * (v.exp() - 1.0) })?
            }
            Node::Softplus(n) => self.unary(&n.inputs[0], &n.outputs[0], values, |v| (1.0 + v.exp()).ln())?,
            Node::Neg(n) => self.unary(&n.inputs[0], &n.outputs[0], values, |v| -v)?,
            Node::Abs(n) => self.unary(&n.inputs[0], &n.outputs[0], values, f32::abs)?,
            Node::Sqrt(n) => self.unary(&n.inputs[0], &n.outputs[0], values, |v| v.max(0.0).sqrt())?,
            Node::Exp(n) => self.unary(&n.inputs[0], &n.outputs[0], values, f32::exp)?,
            Node::Log(n) => self.unary(&n.inputs[0], &n.outputs[0], values, f32::ln)?,
            Node::Clip(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let min_v = match &n.config.min {
                    Some(onnx_ir::node::clip::ClipInput::Static(v)) => *v as f32,
                    Some(onnx_ir::node::clip::ClipInput::Runtime(r)) => {
                        self.resolve_f32_arg(&n.inputs[r.input_index], values)?.data[0]
                    }
                    None => f32::NEG_INFINITY,
                };
                let max_v = match &n.config.max {
                    Some(onnx_ir::node::clip::ClipInput::Static(v)) => *v as f32,
                    Some(onnx_ir::node::clip::ClipInput::Runtime(r)) => {
                        self.resolve_f32_arg(&n.inputs[r.input_index], values)?.data[0]
                    }
                    None => f32::INFINITY,
                };
                values.insert(n.outputs[0].name.clone(), x.map(|v| v.clamp(min_v, max_v)));
            }
            Node::Softmax(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let out = exec_softmax(&x, n.config.axis)?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            Node::Add(n) => {
                let a = self.resolve_f32_arg(&n.inputs[0], values)?;
                let b = self.resolve_f32_arg(&n.inputs[1], values)?;
                let out = broadcast_binary_op(&a, &b, |x, y| x + y)?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            Node::Sub(n) => {
                let a = self.resolve_f32_arg(&n.inputs[0], values)?;
                let b = self.resolve_f32_arg(&n.inputs[1], values)?;
                let out = broadcast_binary_op(&a, &b, |x, y| x - y)?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            Node::Mul(n) => {
                let a = self.resolve_f32_arg(&n.inputs[0], values)?;
                let b = self.resolve_f32_arg(&n.inputs[1], values)?;
                let out = broadcast_binary_op(&a, &b, |x, y| x * y)?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            Node::Div(n) => {
                let a = self.resolve_f32_arg(&n.inputs[0], values)?;
                let b = self.resolve_f32_arg(&n.inputs[1], values)?;
                let out = broadcast_binary_op(&a, &b, |x, y| x / y)?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            Node::Linear(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let w = self.resolve_f32_arg(&n.inputs[1], values)?;
                let b = if n.inputs.len() > 2 && !n.inputs[2].is_optional() {
                    Some(self.resolve_f32_arg(&n.inputs[2], values)?)
                } else {
                    None
                };
                let out = exec_linear(&x, &w, b.as_ref(), n.config.transpose_weight)?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            Node::Gemm(n) => {
                let a = self.resolve_f32_arg(&n.inputs[0], values)?;
                let b = self.resolve_f32_arg(&n.inputs[1], values)?;
                let c = if n.inputs.len() > 2 && !n.inputs[2].is_optional() {
                    Some(self.resolve_f32_arg(&n.inputs[2], values)?)
                } else {
                    None
                };
                let out = exec_gemm(
                    &a,
                    &b,
                    c.as_ref(),
                    n.config.alpha,
                    n.config.beta,
                    n.config.trans_a != 0,
                    n.config.trans_b != 0,
                )?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            Node::MatMul(n) => {
                let a = self.resolve_f32_arg(&n.inputs[0], values)?;
                let b = self.resolve_f32_arg(&n.inputs[1], values)?;
                let out = exec_linear(&a, &b, None, false)?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            Node::Conv1d(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let w = self.resolve_f32_arg(&n.inputs[1], values)?;
                let b = if n.inputs.len() > 2 && !n.inputs[2].is_optional() {
                    Some(self.resolve_f32_arg(&n.inputs[2], values)?)
                } else {
                    None
                };
                let out = exec_conv1d(&x, &w, b.as_ref(), &n.config)?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            Node::BatchNormalization(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let scale = self.resolve_f32_arg(&n.inputs[1], values)?;
                let bias = self.resolve_f32_arg(&n.inputs[2], values)?;
                let mean = self.resolve_f32_arg(&n.inputs[3], values)?;
                let var = self.resolve_f32_arg(&n.inputs[4], values)?;
                let eps = match &n.config {
                    onnx_ir::node::batch_norm::BatchNormConfig::Static(c) => c.epsilon as f32,
                    onnx_ir::node::batch_norm::BatchNormConfig::Runtime(c) => c.epsilon as f32,
                };
                let out = exec_batch_norm(&x, &scale, &bias, &mean, &var, eps)?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            Node::LayerNormalization(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let scale = self.resolve_f32_arg(&n.inputs[1], values)?;
                let bias = if n.inputs.len() > 2 && !n.inputs[2].is_optional() {
                    Some(self.resolve_f32_arg(&n.inputs[2], values)?)
                } else {
                    None
                };
                let out = exec_layer_norm(&x, &scale, bias.as_ref(), n.config.epsilon as f32)?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            Node::Flatten(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let axis = n.config.axis.min(x.shape.len());
                let d0: usize = x.shape[..axis].iter().product::<usize>().max(1);
                let d1: usize = x.shape[axis..].iter().product::<usize>().max(1);
                values.insert(n.outputs[0].name.clone(), x.reshaped(vec![d0, d1])?);
            }
            Node::Reshape(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let raw_shape: Vec<i64> = match &n.config.shape {
                    onnx_ir::node::reshape::ReshapeInput::Static(s) => s.clone(),
                    onnx_ir::node::reshape::ReshapeInput::Runtime(r) => {
                        self.resolve_i64_arg(&n.inputs[r.input_index], i64_values)?
                    }
                };
                let new_shape = resolve_reshape_dims(&x.shape, &raw_shape)?;
                values.insert(n.outputs[0].name.clone(), x.reshaped(new_shape)?);
            }
            Node::Transpose(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let perm: Vec<usize> = n.config.perm.iter().map(|&p| p as usize).collect();
                let out = exec_transpose(&x, &perm)?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            Node::GlobalAveragePool(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                if x.shape.len() != 3 {
                    bail!(
                        "GlobalAveragePool expects rank-3 [N, C, T], got {:?}",
                        x.shape
                    );
                }
                let t3 = x.into_tensor::<3>()?;
                let pooled = t3.global_avg_pool_1d();
                let [b_sz, c_sz] = pooled.shape;
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(vec![b_sz, c_sz, 1], pooled.into_vec(), self.device)?,
                );
            }
            Node::MaxPool1d(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let (kernel, stride) = (n.config.kernel_size, n.config.stride);
                if kernel == 0 || stride == 0 || x.shape.len() != 3 || x.shape[2] < kernel {
                    bail!("MaxPool1d kernel {kernel} / stride {stride} does not fit input {:?}", x.shape);
                }
                let t3 = x.into_tensor::<3>()?;
                let pooled = t3.max_pool1d(kernel, stride);
                values.insert(n.outputs[0].name.clone(), DynTensor::from_tensor(&pooled));
            }
            Node::AveragePool1d(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let out = exec_avg_pool1d(&x, n.config.kernel_size, n.config.stride)?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            Node::Concat(n) => {
                let mut tensors = Vec::with_capacity(n.inputs.len());
                for inp in &n.inputs {
                    tensors.push(self.resolve_f32_arg(inp, values)?);
                }
                let out = exec_concat(&tensors, n.config.axis, self.device)?;
                values.insert(n.outputs[0].name.clone(), out);
            }
            other => {
                bail!("Unsupported ONNX IR node variant in OnnxGraphRunner: {:?}", other.name());
            }
        }
        Ok(())
    }
}

/// A decoded ONNX constant or initializer.
enum Constant {
    F32(DynTensor),
    I64(Vec<i64>),
}

/// Decodes `arg`'s constant value: integer data stays `i64` (shapes, indices), everything else
/// becomes `f32`.
fn decode_constant(arg: &Argument, device: SynapseMlDevice) -> Option<Constant> {
    let td = arg.value()?;
    if let Ok(v) = td.to_vec::<f32>() {
        return DynTensor::new(td.shape.to_vec(), v, device).ok().map(Constant::F32);
    }
    if let Ok(v) = td.to_vec::<i64>() {
        return Some(Constant::I64(v));
    }
    td.clone().convert::<f32>().to_vec::<f32>().ok().and_then(|v| DynTensor::new(td.shape.to_vec(), v, device).ok()).map(Constant::F32)
}

fn port_spec_from_arg(arg: &Argument) -> OnnxPortSpec {
    match &arg.ty {
        ArgType::Tensor(t) => OnnxPortSpec {
            name: arg.name.clone(),
            rank: t.rank,
            static_shape: t.static_shape.clone(),
        },
        ArgType::Shape(r) => OnnxPortSpec {
            name: arg.name.clone(),
            rank: *r,
            static_shape: None,
        },
        ArgType::ScalarTensor(_) | ArgType::ScalarNative(_) => OnnxPortSpec {
            name: arg.name.clone(),
            rank: 0,
            static_shape: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_graph_runner_gemm_relu_from_bytes() {
        let dev = SynapseMlDevice::Cpu;
        // W is [out=2, in=3]:
        // out 0 = 1*x0 + 2*x1 + 3*x2 - 1.0
        // out 1 = -1*x0 - 2*x1 - 3*x2 + 0.5
        let w = vec![1.0f32, 2.0, 3.0, -1.0, -2.0, -3.0];
        let b = vec![-1.0f32, 0.5];
        let onnx_bytes = onnx_proto_builder::encode_linear_relu_onnx_bytes(
            "input", "output", 3, 2, &w, &b, true,
        );

        let runner = OnnxGraphRunner::from_bytes(&onnx_bytes, dev).unwrap();
        assert_eq!(runner.input_specs().len(), 1);
        assert_eq!(runner.output_specs().len(), 1);

        let x = Tensor2D::from_floats(vec![1.0, 1.0, 1.0], [1, 3], dev);
        let y = runner.run_2d(&x).unwrap();
        assert_eq!(y.shape, [1, 2]);
        // out 0 = (1+2+3 - 1) = 5.0 -> ReLU(5.0) = 5.0
        // out 1 = (-1-2-3 + 0.5) = -5.5 -> ReLU(-5.5) = 0.0
        assert!((y.data[0] - 5.0).abs() < 1e-5);
        assert!((y.data[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn weights_are_decoded_once_and_shared_across_runs() {
        let dev = SynapseMlDevice::Cpu;
        let (w, b) = (vec![1.0f32, 2.0, 3.0, -1.0, -2.0, -3.0], vec![-1.0f32, 0.5]);
        let bytes = onnx_proto_builder::encode_linear_relu_onnx_bytes("input", "output", 3, 2, &w, &b, true);
        let runner = OnnxGraphRunner::from_bytes(&bytes, dev).unwrap();
        assert!(!runner.constants.is_empty(), "initializers are cached at load");

        let x = Tensor2D::from_floats(vec![1.0, 1.0, 1.0], [1, 3], dev);
        let first = runner.run_2d(&x).unwrap();
        for _ in 0..3 {
            assert_eq!(runner.run_2d(&x).unwrap().data, first.data);
        }
        // Every lookup of a weight hands out the buffer decoded at load
        let empty = HashMap::new();
        for node in &runner.graph.nodes {
            for arg in node.inputs().iter().filter(|a| runner.constants.contains_key(&a.name)) {
                let a = runner.resolve_f32_arg(arg, &empty).unwrap();
                let b = runner.resolve_f32_arg(arg, &empty).unwrap();
                assert!(std::sync::Arc::ptr_eq(&a.data, &b.data));
                assert!(std::sync::Arc::ptr_eq(&a.data, &runner.constants[&arg.name].data));
            }
        }
    }
}
