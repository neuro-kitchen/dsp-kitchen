//! `OnnxGraphRunner`: Zero-copy `.onnx` graph parser (`onnx-ir = "0.21.0"`) and
//! dynamic Burn tensor execution engine for running external spike-sorting models
//! (Kilosort4, DARTsort, CEBRA, Bombcell/UnitMatch) inside `dsp-synapse-ml`.

use std::collections::HashMap;
use std::path::Path;
use anyhow::{Context, Result, bail};
use onnx_ir::ir::{ArgType, Argument, Node, OnnxGraph};
use onnx_ir::node::padding::PaddingConfig1d;
use onnx_ir::OnnxGraphBuilder;
use serde::{Deserialize, Serialize};
use crate::backend::{SynapseMlDevice, Tensor, Tensor1D, Tensor2D, Tensor3D};
use crate::hub::transpose_2d_slice;

/// Metadata describing a graph input or output port in an [`OnnxGraphRunner`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnnxPortSpec {
    pub name: String,
    pub rank: usize,
    pub static_shape: Option<Vec<Option<usize>>>,
}

/// Dynamic-rank `f32` tensor value flowing through [`OnnxGraphRunner`].
#[derive(Debug, Clone, PartialEq)]
pub struct DynTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub device: SynapseMlDevice,
}

impl DynTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>, device: SynapseMlDevice) -> Result<Self> {
        let expected: usize = shape.iter().product::<usize>().max(if shape.is_empty() { 1 } else { 0 });
        if !shape.is_empty() && data.len() != shape.iter().product::<usize>() {
            bail!(
                "DynTensor shape {:?} expects {} elements, got {}",
                shape,
                shape.iter().product::<usize>(),
                data.len()
            );
        }
        if shape.is_empty() && data.len() != 1 {
            bail!("Scalar DynTensor expects 1 element, got {}", data.len());
        }
        let _ = expected;
        Ok(Self {
            shape,
            data,
            device,
        })
    }

    pub fn from_tensor<const D: usize>(t: &Tensor<D>) -> Self {
        Self {
            shape: t.shape.to_vec(),
            data: t.data.clone(),
            device: t.device,
        }
    }

    pub fn into_tensor<const D: usize>(self) -> Result<Tensor<D>> {
        if self.shape.len() != D {
            bail!(
                "Expected rank-{} tensor, got rank-{} with shape {:?}",
                D,
                self.shape.len(),
                self.shape
            );
        }
        let mut arr = [0usize; D];
        arr.copy_from_slice(&self.shape);
        Ok(Tensor::from_floats(self.data, arr, self.device))
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
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

    /// Wraps a pre-parsed [`onnx_ir::OnnxGraph`].
    pub fn from_graph(graph: OnnxGraph, device: SynapseMlDevice) -> Self {
        let inputs = graph.inputs.iter().map(port_spec_from_arg).collect();
        let outputs = graph.outputs.iter().map(port_spec_from_arg).collect();
        Self {
            graph,
            device,
            inputs,
            outputs,
        }
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

    fn resolve_f32_arg(
        &self,
        arg: &Argument,
        values: &HashMap<String, DynTensor>,
    ) -> Result<DynTensor> {
        if !arg.name.is_empty()
            && let Some(v) = values.get(&arg.name)
        {
            return Ok(v.clone());
        }
        if let Some(td) = arg.value() {
            let shape = td.shape.to_vec();
            let vec: Vec<f32> = td
                .to_vec::<f32>()
                .or_else(|_| td.clone().convert::<f32>().to_vec::<f32>())
                .map_err(|e| anyhow::anyhow!("Failed converting ONNX constant '{}' to f32: {:?}", arg.name, e))?;
            return DynTensor::new(shape, vec, self.device);
        }
        bail!("Missing runtime or constant f32 tensor for argument '{}'", arg.name)
    }

    fn resolve_i64_arg(
        &self,
        arg: &Argument,
        i64_values: &HashMap<String, Vec<i64>>,
    ) -> Result<Vec<i64>> {
        if !arg.name.is_empty()
            && let Some(v) = i64_values.get(&arg.name)
        {
            return Ok(v.clone());
        }
        if let Some(td) = arg.value() {
            let vec: Vec<i64> = td
                .to_vec::<i64>()
                .or_else(|_| td.clone().convert::<i64>().to_vec::<i64>())
                .map_err(|e| anyhow::anyhow!("Failed converting ONNX constant '{}' to i64: {:?}", arg.name, e))?;
            return Ok(vec);
        }
        bail!("Missing i64 tensor/shape for argument '{}'", arg.name)
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
                if let Some(td) = src_arg.value().or_else(|| out_arg.value()) {
                    if let Ok(vec_f32) = td.to_vec::<f32>() {
                        let t = DynTensor::new(td.shape.to_vec(), vec_f32, self.device)?;
                        values.insert(out_arg.name.clone(), t);
                    } else if let Ok(vec_i64) = td.to_vec::<i64>() {
                        i64_values.insert(out_arg.name.clone(), vec_i64);
                    } else if let Ok(vec_f32) = td.clone().convert::<f32>().to_vec::<f32>() {
                        let t = DynTensor::new(td.shape.to_vec(), vec_f32, self.device)?;
                        values.insert(out_arg.name.clone(), t);
                    }
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
            Node::Relu(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let data = x.data.iter().map(|&v| v.max(0.0)).collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
            }
            Node::LeakyRelu(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let alpha = n.config.alpha as f32;
                let data = x
                    .data
                    .iter()
                    .map(|&v| if v >= 0.0 { v } else { alpha * v })
                    .collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
            }
            Node::Gelu(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                let data = x
                    .data
                    .iter()
                    .map(|&v| {
                        let inner = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v);
                        0.5 * v * (1.0 + inner.tanh())
                    })
                    .collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
            }
            Node::Sigmoid(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let data = x.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
            }
            Node::Tanh(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let data = x.data.iter().map(|&v| v.tanh()).collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
            }
            Node::Swish(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let data = x
                    .data
                    .iter()
                    .map(|&v| v / (1.0 + (-v).exp()))
                    .collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
            }
            Node::Elu(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let alpha = n.config.alpha as f32;
                let data = x
                    .data
                    .iter()
                    .map(|&v| if v >= 0.0 { v } else { alpha * (v.exp() - 1.0) })
                    .collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
            }
            Node::Softplus(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let data = x.data.iter().map(|&v| (1.0 + v.exp()).ln()).collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
            }
            Node::Neg(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let data = x.data.iter().map(|&v| -v).collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
            }
            Node::Abs(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let data = x.data.iter().map(|&v| v.abs()).collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
            }
            Node::Sqrt(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let data = x.data.iter().map(|&v| v.max(0.0).sqrt()).collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
            }
            Node::Exp(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let data = x.data.iter().map(|&v| v.exp()).collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
            }
            Node::Log(n) => {
                let x = self.resolve_f32_arg(&n.inputs[0], values)?;
                let data = x.data.iter().map(|&v| v.ln()).collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
            }
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
                let data = x.data.iter().map(|&v| v.clamp(min_v, max_v)).collect();
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(x.shape, data, self.device)?,
                );
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
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(vec![d0, d1], x.data, self.device)?,
                );
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
                values.insert(
                    n.outputs[0].name.clone(),
                    DynTensor::new(new_shape, x.data, self.device)?,
                );
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
                let t3 = x.into_tensor::<3>()?;
                let pooled = t3.max_pool1d(n.config.kernel_size, n.config.stride);
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

fn exec_linear(
    x: &DynTensor,
    w: &DynTensor,
    bias: Option<&DynTensor>,
    transpose_weight: bool,
) -> Result<DynTensor> {
    if x.shape.is_empty() || w.shape.len() != 2 {
        bail!(
            "Linear expects input rank >= 1 and 2D weight, got input {:?} and weight {:?}",
            x.shape,
            w.shape
        );
    }
    let in_features = *x.shape.last().unwrap();
    let batch_rows: usize = x.shape[..x.shape.len() - 1].iter().product::<usize>().max(1);

    // If transpose_weight is true (PyTorch / ONNX Gemm transB=1 convention), W is [out_features, in_features].
    // Otherwise W is [in_features, out_features].
    let (w_in_out, out_features) = if transpose_weight {
        let [out_f, in_f] = [w.shape[0], w.shape[1]];
        if in_f != in_features {
            bail!(
                "Linear(transpose_weight=true) feature mismatch: input has {}, weight is [{}, {}]",
                in_features,
                out_f,
                in_f
            );
        }
        let transposed = transpose_2d_slice(&w.data, out_f, in_f);
        (
            Tensor2D::from_floats(transposed, [in_f, out_f], x.device),
            out_f,
        )
    } else {
        let [in_f, out_f] = [w.shape[0], w.shape[1]];
        if in_f != in_features {
            bail!(
                "Linear(transpose_weight=false) feature mismatch: input has {}, weight is [{}, {}]",
                in_features,
                in_f,
                out_f
            );
        }
        (
            Tensor2D::from_floats(w.data.clone(), [in_f, out_f], x.device),
            out_f,
        )
    };

    let x_2d = Tensor2D::from_floats(x.data.clone(), [batch_rows, in_features], x.device);
    let mut y_2d = x_2d.matmul(&w_in_out);

    if let Some(b) = bias {
        if b.numel() == out_features {
            let b_1d = Tensor1D::from_floats(b.data.clone(), [out_features], x.device);
            y_2d = y_2d.add_bias_1d(&b_1d);
        } else if b.numel() == 1 {
            let s = b.data[0];
            for v in &mut y_2d.data {
                *v += s;
            }
        } else {
            bail!(
                "Linear bias length {} does not match out_features {}",
                b.numel(),
                out_features
            );
        }
    }

    let mut out_shape = x.shape[..x.shape.len() - 1].to_vec();
    out_shape.push(out_features);
    DynTensor::new(out_shape, y_2d.into_vec(), x.device)
}

fn exec_gemm(
    a: &DynTensor,
    b: &DynTensor,
    c: Option<&DynTensor>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) -> Result<DynTensor> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        bail!("Gemm requires rank-2 matrices, got {:?} and {:?}", a.shape, b.shape);
    }
    let a_mat = if trans_a {
        let [r, col] = [a.shape[0], a.shape[1]];
        Tensor2D::from_floats(transpose_2d_slice(&a.data, r, col), [col, r], a.device)
    } else {
        Tensor2D::from_floats(a.data.clone(), [a.shape[0], a.shape[1]], a.device)
    };

    let b_mat = if trans_b {
        let [r, col] = [b.shape[0], b.shape[1]];
        Tensor2D::from_floats(transpose_2d_slice(&b.data, r, col), [col, r], b.device)
    } else {
        Tensor2D::from_floats(b.data.clone(), [b.shape[0], b.shape[1]], b.device)
    };

    let mut y = a_mat.matmul(&b_mat);
    if (alpha - 1.0).abs() > 1e-7 {
        y = y.mul_scalar(alpha);
    }

    if let Some(c_tensor) = c
        && beta.abs() > 0.0
    {
        let [m, n] = y.shape;
        if c_tensor.numel() == n {
            for r in 0..m {
                for col in 0..n {
                    y.data[r * n + col] += beta * c_tensor.data[col];
                }
            }
        } else if c_tensor.numel() == m * n {
            for i in 0..(m * n) {
                y.data[i] += beta * c_tensor.data[i];
            }
        } else if c_tensor.numel() == 1 {
            let s = beta * c_tensor.data[0];
            for v in &mut y.data {
                *v += s;
            }
        } else {
            bail!(
                "Unsupported Gemm C broadcast shape {:?} for output {:?}",
                c_tensor.shape,
                y.shape
            );
        }
    }

    Ok(DynTensor::from_tensor(&y))
}

fn exec_conv1d(
    x: &DynTensor,
    w: &DynTensor,
    bias: Option<&DynTensor>,
    cfg: &onnx_ir::node::conv1d::Conv1dConfig,
) -> Result<DynTensor> {
    if x.shape.len() != 3 || w.shape.len() != 3 {
        bail!(
            "Conv1d expects 3D input [N, C_in, L] and 3D weight [C_out, C_in/groups, K], got {:?} and {:?}",
            x.shape,
            w.shape
        );
    }
    let [batch_size, c_in, l_in] = [x.shape[0], x.shape[1], x.shape[2]];
    let [c_out, c_in_per_group, k_size] = [w.shape[0], w.shape[1], w.shape[2]];
    let groups = cfg.groups.max(1);
    let stride = cfg.stride.max(1);
    let dilation = cfg.dilation.max(1);

    if c_in != c_in_per_group * groups {
        bail!(
            "Conv1d channel mismatch: c_in={} vs c_in_per_group={} * groups={}",
            c_in,
            c_in_per_group,
            groups
        );
    }

    let (pad_left, pad_right) = match cfg.padding {
        PaddingConfig1d::Valid => (0usize, 0usize),
        PaddingConfig1d::Explicit(l, r) => (l, r),
    };

    // Fast path for standard 1D convolution (groups == 1, dilation == 1, symmetric padding)
    if groups == 1 && dilation == 1 && pad_left == pad_right {
        let bias_1d = match bias {
            Some(b) => Tensor1D::from_floats(b.data.clone(), [c_out], x.device),
            None => Tensor1D::zeros([c_out], x.device),
        };
        let layer = crate::backbones::Conv1dLayer {
            weight: Tensor3D::from_floats(w.data.clone(), [c_out, c_in, k_size], x.device),
            bias: bias_1d,
            stride,
            padding: pad_left,
        };
        let x_3d = Tensor3D::from_floats(x.data.clone(), [batch_size, c_in, l_in], x.device);
        let out_3d = layer.forward(&x_3d);
        return Ok(DynTensor::from_tensor(&out_3d));
    }

    // General path supporting arbitrary dilation, groups, and asymmetric padding
    let eff_k = (k_size - 1) * dilation + 1;
    let padded_len = l_in + pad_left + pad_right;
    if padded_len < eff_k {
        bail!(
            "Conv1d padded length {} is smaller than dilated kernel {}",
            padded_len,
            eff_k
        );
    }
    let l_out = (padded_len - eff_k) / stride + 1;
    let c_out_per_group = c_out / groups;
    let mut out = vec![0.0f32; batch_size * c_out * l_out];

    for b in 0..batch_size {
        for g in 0..groups {
            for co_g in 0..c_out_per_group {
                let co = g * c_out_per_group + co_g;
                let b_val = bias.map(|bt| bt.data[co]).unwrap_or(0.0);
                for o in 0..l_out {
                    let in_start = (o * stride) as isize - (pad_left as isize);
                    let mut acc = b_val;
                    for ci_g in 0..c_in_per_group {
                        let ci = g * c_in_per_group + ci_g;
                        let x_off = (b * c_in + ci) * l_in;
                        let w_off = (co * c_in_per_group + ci_g) * k_size;
                        for k in 0..k_size {
                            let pos = in_start + (k * dilation) as isize;
                            if pos >= 0 && (pos as usize) < l_in {
                                acc += x.data[x_off + (pos as usize)] * w.data[w_off + k];
                            }
                        }
                    }
                    out[(b * c_out + co) * l_out + o] = acc;
                }
            }
        }
    }

    DynTensor::new(vec![batch_size, c_out, l_out], out, x.device)
}

fn exec_batch_norm(
    x: &DynTensor,
    scale: &DynTensor,
    bias: &DynTensor,
    mean: &DynTensor,
    var: &DynTensor,
    eps: f32,
) -> Result<DynTensor> {
    if x.shape.len() < 2 {
        bail!("BatchNormalization expects rank >= 2, got {:?}", x.shape);
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; x.data.len()];

    for ch in 0..c {
        let m = mean.data[ch];
        let inv_std = 1.0 / (var.data[ch] + eps).sqrt();
        let sc = scale.data[ch] * inv_std;
        let sh = bias.data[ch] - m * sc;
        for b in 0..n {
            let off = (b * c + ch) * spatial;
            for s in 0..spatial {
                out[off + s] = x.data[off + s] * sc + sh;
            }
        }
    }
    DynTensor::new(x.shape.clone(), out, x.device)
}

fn exec_layer_norm(
    x: &DynTensor,
    scale: &DynTensor,
    bias: Option<&DynTensor>,
    eps: f32,
) -> Result<DynTensor> {
    if x.shape.is_empty() {
        bail!("LayerNormalization requires rank >= 1");
    }
    let norm_dim = scale.numel();
    let outer = x.numel() / norm_dim.max(1);
    if outer * norm_dim != x.numel() {
        bail!(
            "LayerNormalization scale size {} does not divide input shape {:?}",
            norm_dim,
            x.shape
        );
    }
    let mut out = vec![0.0f32; x.numel()];
    let inv_d = 1.0 / (norm_dim as f32);

    for r in 0..outer {
        let row = &x.data[r * norm_dim..(r + 1) * norm_dim];
        let mean: f32 = row.iter().sum::<f32>() * inv_d;
        let var: f32 = row
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            * inv_d;
        let inv_std = 1.0 / (var + eps).sqrt();
        for c in 0..norm_dim {
            let b_val = bias.map(|b| b.data[c]).unwrap_or(0.0);
            out[r * norm_dim + c] = (row[c] - mean) * inv_std * scale.data[c] + b_val;
        }
    }
    DynTensor::new(x.shape.clone(), out, x.device)
}

fn exec_softmax(x: &DynTensor, axis: usize) -> Result<DynTensor> {
    if axis >= x.shape.len() {
        bail!("Softmax axis {} out of bounds for shape {:?}", axis, x.shape);
    }
    let outer: usize = x.shape[..axis].iter().product::<usize>().max(1);
    let dim = x.shape[axis];
    let inner: usize = x.shape[axis + 1..].iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; x.numel()];

    for o in 0..outer {
        for i in 0..inner {
            let mut max_v = f32::NEG_INFINITY;
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                if x.data[idx] > max_v {
                    max_v = x.data[idx];
                }
            }
            let mut sum = 0.0f32;
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                let e = (x.data[idx] - max_v).exp();
                out[idx] = e;
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-12);
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                out[idx] *= inv;
            }
        }
    }
    DynTensor::new(x.shape.clone(), out, x.device)
}

fn broadcast_binary_op<F>(a: &DynTensor, b: &DynTensor, op: F) -> Result<DynTensor>
where
    F: Fn(f32, f32) -> f32,
{
    if a.shape == b.shape {
        let data = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| op(x, y))
            .collect();
        return DynTensor::new(a.shape.clone(), data, a.device);
    }
    if b.numel() == 1 {
        let s = b.data[0];
        let data = a.data.iter().map(|&x| op(x, s)).collect();
        return DynTensor::new(a.shape.clone(), data, a.device);
    }
    if a.numel() == 1 {
        let s = a.data[0];
        let data = b.data.iter().map(|&y| op(s, y)).collect();
        return DynTensor::new(b.shape.clone(), data, b.device);
    }

    // General N-D NumPy/ONNX right-aligned broadcasting
    let rank = a.shape.len().max(b.shape.len());
    let mut out_shape = vec![1usize; rank];
    let mut a_padded = vec![1usize; rank];
    let mut b_padded = vec![1usize; rank];

    for i in 0..a.shape.len() {
        a_padded[rank - a.shape.len() + i] = a.shape[i];
    }
    for i in 0..b.shape.len() {
        b_padded[rank - b.shape.len() + i] = b.shape[i];
    }
    for i in 0..rank {
        if a_padded[i] == b_padded[i] {
            out_shape[i] = a_padded[i];
        } else if a_padded[i] == 1 {
            out_shape[i] = b_padded[i];
        } else if b_padded[i] == 1 {
            out_shape[i] = a_padded[i];
        } else {
            bail!(
                "Incompatible broadcast shapes {:?} and {:?}",
                a.shape,
                b.shape
            );
        }
    }

    let out_strides = compute_strides(&out_shape);
    let a_strides = compute_strides(&a_padded);
    let b_strides = compute_strides(&b_padded);
    let total: usize = out_shape.iter().product();
    let mut out_data = vec![0.0f32; total];

    for flat in 0..total {
        let mut rem = flat;
        let mut a_idx = 0usize;
        let mut b_idx = 0usize;
        for d in 0..rank {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            let ac = if a_padded[d] == 1 { 0 } else { coord };
            let bc = if b_padded[d] == 1 { 0 } else { coord };
            a_idx += ac * a_strides[d];
            b_idx += bc * b_strides[d];
        }
        out_data[flat] = op(a.data[a_idx], b.data[b_idx]);
    }

    DynTensor::new(out_shape, out_data, a.device)
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1].max(1);
    }
    strides
}

fn resolve_reshape_dims(input_shape: &[usize], target: &[i64]) -> Result<Vec<usize>> {
    let total: usize = input_shape.iter().product();
    let mut out = vec![0usize; target.len()];
    let mut infer_idx: Option<usize> = None;
    let mut known_prod = 1usize;

    for (i, &dim) in target.iter().enumerate() {
        if dim == -1 {
            if infer_idx.is_some() {
                bail!("Reshape target {:?} has multiple -1 dimensions", target);
            }
            infer_idx = Some(i);
        } else if dim == 0 {
            let copied = *input_shape
                .get(i)
                .with_context(|| format!("Reshape 0-dim at {} out of bounds for {:?}", i, input_shape))?;
            out[i] = copied;
            known_prod *= copied;
        } else if dim > 0 {
            out[i] = dim as usize;
            known_prod *= dim as usize;
        } else {
            bail!("Invalid Reshape dimension {}", dim);
        }
    }

    if let Some(idx) = infer_idx {
        if known_prod == 0 || total % known_prod != 0 {
            bail!(
                "Cannot infer -1 dimension for total {} with known product {}",
                total,
                known_prod
            );
        }
        out[idx] = total / known_prod;
    }

    Ok(out)
}

fn exec_transpose(x: &DynTensor, perm: &[usize]) -> Result<DynTensor> {
    let rank = x.shape.len();
    if perm.len() != rank {
        bail!(
            "Transpose perm {:?} does not match input rank {} ({:?})",
            perm,
            rank,
            x.shape
        );
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| x.shape[p]).collect();
    let in_strides = compute_strides(&x.shape);
    let out_strides = compute_strides(&out_shape);
    let total = x.numel();
    let mut out = vec![0.0f32; total];

    for flat_in in 0..total {
        let mut rem = flat_in;
        let mut flat_out = 0usize;
        for d in 0..rank {
            let coord = rem / in_strides[d];
            rem %= in_strides[d];
            // Find where dimension d lands in perm
            for (out_d, &p) in perm.iter().enumerate() {
                if p == d {
                    flat_out += coord * out_strides[out_d];
                    break;
                }
            }
        }
        out[flat_out] = x.data[flat_in];
    }

    DynTensor::new(out_shape, out, x.device)
}

fn exec_avg_pool1d(x: &DynTensor, kernel_size: usize, stride: usize) -> Result<DynTensor> {
    if x.shape.len() != 3 {
        bail!("AveragePool1d expects 3D [N, C, T], got {:?}", x.shape);
    }
    let [n, c, t] = [x.shape[0], x.shape[1], x.shape[2]];
    if t < kernel_size || kernel_size == 0 || stride == 0 {
        return Ok(x.clone());
    }
    let t_out = (t - kernel_size) / stride + 1;
    let inv_k = 1.0 / (kernel_size as f32);
    let mut out = vec![0.0f32; n * c * t_out];

    for b in 0..n {
        for ch in 0..c {
            let in_off = (b * c + ch) * t;
            let out_off = (b * c + ch) * t_out;
            for o in 0..t_out {
                let s = o * stride;
                let sum: f32 = x.data[in_off + s..in_off + s + kernel_size].iter().sum();
                out[out_off + o] = sum * inv_k;
            }
        }
    }
    DynTensor::new(vec![n, c, t_out], out, x.device)
}

fn exec_concat(
    tensors: &[DynTensor],
    axis: usize,
    device: SynapseMlDevice,
) -> Result<DynTensor> {
    let first = tensors.first().context("Concat requires at least 1 input")?;
    let rank = first.shape.len();
    if axis >= rank {
        bail!("Concat axis {} out of bounds for rank {}", axis, rank);
    }

    let mut out_shape = first.shape.clone();
    out_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();

    let outer: usize = first.shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = first.shape[axis + 1..].iter().product::<usize>().max(1);
    let mut out_data = Vec::with_capacity(out_shape.iter().product());

    for o in 0..outer {
        for t in tensors {
            let dim_t = t.shape[axis];
            let chunk = dim_t * inner;
            let start = o * chunk;
            out_data.extend_from_slice(&t.data[start..start + chunk]);
        }
    }

    DynTensor::new(out_shape, out_data, device)
}

/// Builder helper for constructing valid serialized `.onnx` (`ModelProto`) byte buffers
/// in memory (used by unit tests and lightweight model exporters without requiring external `.onnx` files).
pub mod onnx_proto_builder {
    /// Encodes a minimal valid ONNX `ModelProto` (opset 17, IR version 8) containing a single
    /// `Gemm` + `Relu` + `Gemm` MLP or single `Gemm` projection so [`super::OnnxGraphRunner::from_bytes`]
    /// exercises the full `onnx_ir::OnnxGraphBuilder` protobuf parser and simplification pipeline.
    pub fn encode_linear_relu_onnx_bytes(
        input_name: &str,
        output_name: &str,
        in_features: usize,
        out_features: usize,
        weight_out_in: &[f32],
        bias_out: &[f32],
        with_relu: bool,
    ) -> Vec<u8> {
        assert_eq!(weight_out_in.len(), out_features * in_features);
        assert_eq!(bias_out.len(), out_features);

        // We use minimal raw Protobuf wire encoding for ONNX ModelProto -> GraphProto ->
        // ValueInfoProto / TensorProto / NodeProto so we don't depend on internal protobuf private types.
        let w_init = encode_tensor_proto("W", &[out_features as i64, in_features as i64], weight_out_in);
        let b_init = encode_tensor_proto("B", &[out_features as i64], bias_out);

        let gemm_out_name = if with_relu { "gemm_out" } else { output_name };
        let gemm_node = encode_gemm_node("gemm_0", input_name, "W", "B", gemm_out_name, 1);

        let mut graph_bytes = Vec::new();
        // NodeProto = field 1 in GraphProto
        write_bytes_field(&mut graph_bytes, 1, &gemm_node);
        if with_relu {
            let relu_node = encode_unary_node("relu_0", "Relu", gemm_out_name, output_name);
            write_bytes_field(&mut graph_bytes, 1, &relu_node);
        }
        // name = field 2 in GraphProto
        write_bytes_field(&mut graph_bytes, 2, b"synapse_ml_onnx_graph");
        // initializer = field 5 in GraphProto
        write_bytes_field(&mut graph_bytes, 5, &w_init);
        write_bytes_field(&mut graph_bytes, 5, &b_init);
        // input = field 11 in GraphProto
        let in_vi = encode_value_info_f32(input_name, &[0, in_features as i64]);
        write_bytes_field(&mut graph_bytes, 11, &in_vi);
        // output = field 12 in GraphProto
        let out_vi = encode_value_info_f32(output_name, &[0, out_features as i64]);
        write_bytes_field(&mut graph_bytes, 12, &out_vi);

        let mut model_bytes = Vec::new();
        // ir_version = field 1 (varint = 8)
        write_varint_field(&mut model_bytes, 1, 8);
        // graph = field 7
        write_bytes_field(&mut model_bytes, 7, &graph_bytes);
        // opset_import = field 8 (OperatorSetIdProto { domain: "", version: 17 })
        let mut opset = Vec::new();
        write_bytes_field(&mut opset, 1, b"");
        write_varint_field(&mut opset, 2, 17);
        write_bytes_field(&mut model_bytes, 8, &opset);

        model_bytes
    }

    /// Encodes a 3D `[N, K, T]` -> `Flatten(axis=1)` -> `Gemm` -> `[N, D]` ONNX `ModelProto`.
    pub fn encode_flatten_gemm_onnx_bytes(
        input_name: &str,
        output_name: &str,
        num_channels: usize,
        num_samples: usize,
        out_features: usize,
        weight_out_in: &[f32],
        bias_out: &[f32],
    ) -> Vec<u8> {
        let in_features = num_channels * num_samples;
        assert_eq!(weight_out_in.len(), out_features * in_features);
        assert_eq!(bias_out.len(), out_features);

        let w_init = encode_tensor_proto("W", &[out_features as i64, in_features as i64], weight_out_in);
        let b_init = encode_tensor_proto("B", &[out_features as i64], bias_out);

        let flatten_node = encode_flatten_node("flatten_0", input_name, "flat_out", 1);
        let gemm_node = encode_gemm_node("gemm_0", "flat_out", "W", "B", output_name, 1);

        let mut graph_bytes = Vec::new();
        write_bytes_field(&mut graph_bytes, 1, &flatten_node);
        write_bytes_field(&mut graph_bytes, 1, &gemm_node);
        write_bytes_field(&mut graph_bytes, 2, b"synapse_ml_3d_to_2d_graph");
        write_bytes_field(&mut graph_bytes, 5, &w_init);
        write_bytes_field(&mut graph_bytes, 5, &b_init);

        let in_vi = encode_value_info_f32(
            input_name,
            &[0, num_channels as i64, num_samples as i64],
        );
        write_bytes_field(&mut graph_bytes, 11, &in_vi);
        let out_vi = encode_value_info_f32(output_name, &[0, out_features as i64]);
        write_bytes_field(&mut graph_bytes, 12, &out_vi);

        let mut model_bytes = Vec::new();
        write_varint_field(&mut model_bytes, 1, 8);
        write_bytes_field(&mut model_bytes, 7, &graph_bytes);
        let mut opset = Vec::new();
        write_bytes_field(&mut opset, 1, b"");
        write_varint_field(&mut opset, 2, 17);
        write_bytes_field(&mut model_bytes, 8, &opset);
        model_bytes
    }

    /// Encodes a 3D `[N, C, T]` -> `Conv1d(kernel=3, pad=1)` -> `Relu` -> `[N, C, T]` ONNX `ModelProto`.
    pub fn encode_conv1d_relu_onnx_bytes(
        input_name: &str,
        output_name: &str,
        channels: usize,
        samples: usize,
        kernel_size: usize,
        padding: usize,
        weight_oik: &[f32],
        bias_out: &[f32],
    ) -> Vec<u8> {
        assert_eq!(weight_oik.len(), channels * channels * kernel_size);
        assert_eq!(bias_out.len(), channels);

        let w_init = encode_tensor_proto(
            "W",
            &[channels as i64, channels as i64, kernel_size as i64],
            weight_oik,
        );
        let b_init = encode_tensor_proto("B", &[channels as i64], bias_out);

        let conv_node = encode_conv1d_node(
            "conv_0",
            input_name,
            "W",
            "B",
            "conv_out",
            kernel_size as i64,
            1,
            padding as i64,
        );
        let relu_node = encode_unary_node("relu_0", "Relu", "conv_out", output_name);

        let mut graph_bytes = Vec::new();
        write_bytes_field(&mut graph_bytes, 1, &conv_node);
        write_bytes_field(&mut graph_bytes, 1, &relu_node);
        write_bytes_field(&mut graph_bytes, 2, b"synapse_ml_conv1d_graph");
        write_bytes_field(&mut graph_bytes, 5, &w_init);
        write_bytes_field(&mut graph_bytes, 5, &b_init);

        let in_vi = encode_value_info_f32(input_name, &[0, channels as i64, samples as i64]);
        write_bytes_field(&mut graph_bytes, 11, &in_vi);
        let out_vi = encode_value_info_f32(output_name, &[0, channels as i64, samples as i64]);
        write_bytes_field(&mut graph_bytes, 12, &out_vi);

        let mut model_bytes = Vec::new();
        write_varint_field(&mut model_bytes, 1, 8);
        write_bytes_field(&mut model_bytes, 7, &graph_bytes);
        let mut opset = Vec::new();
        write_bytes_field(&mut opset, 1, b"");
        write_varint_field(&mut opset, 2, 17);
        write_bytes_field(&mut model_bytes, 8, &opset);
        model_bytes
    }

    fn write_varint(buf: &mut Vec<u8>, mut val: u64) {
        loop {
            if (val & !0x7Fu64) == 0 {
                buf.push(val as u8);
                break;
            } else {
                buf.push(((val & 0x7F) | 0x80) as u8);
                val >>= 7;
            }
        }
    }

    fn write_varint_field(buf: &mut Vec<u8>, field_num: u32, val: u64) {
        write_varint(buf, ((field_num as u64) << 3) | 0);
        write_varint(buf, val);
    }

    fn write_bytes_field(buf: &mut Vec<u8>, field_num: u32, payload: &[u8]) {
        write_varint(buf, ((field_num as u64) << 3) | 2);
        write_varint(buf, payload.len() as u64);
        buf.extend_from_slice(payload);
    }

    fn encode_tensor_proto(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
        let mut buf = Vec::new();
        for &d in dims {
            write_varint_field(&mut buf, 1, d as u64);
        }
        // data_type = 1 (FLOAT)
        write_varint_field(&mut buf, 2, 1);
        // name = field 8
        write_bytes_field(&mut buf, 8, name.as_bytes());
        // raw_data = field 9
        let raw_bytes: &[u8] = bytemuck::cast_slice(data);
        write_bytes_field(&mut buf, 9, raw_bytes);
        buf
    }

    fn encode_value_info_f32(name: &str, dims: &[i64]) -> Vec<u8> {
        let mut shape_proto = Vec::new();
        for &d in dims {
            let mut dim_proto = Vec::new();
            if d > 0 {
                write_varint_field(&mut dim_proto, 1, d as u64);
            } else {
                write_bytes_field(&mut dim_proto, 2, b"batch");
            }
            write_bytes_field(&mut shape_proto, 1, &dim_proto);
        }

        let mut tensor_type = Vec::new();
        // elem_type = 1 (FLOAT)
        write_varint_field(&mut tensor_type, 1, 1);
        // shape = field 2
        write_bytes_field(&mut tensor_type, 2, &shape_proto);

        let mut type_proto = Vec::new();
        // tensor_type = field 1
        write_bytes_field(&mut type_proto, 1, &tensor_type);

        let mut vi = Vec::new();
        write_bytes_field(&mut vi, 1, name.as_bytes());
        write_bytes_field(&mut vi, 2, &type_proto);
        vi
    }

    fn encode_attr_int(name: &str, val: i64) -> Vec<u8> {
        let mut attr = Vec::new();
        write_bytes_field(&mut attr, 1, name.as_bytes());
        write_varint_field(&mut attr, 3, val as u64);
        // AttributeType::INT = 2
        write_varint_field(&mut attr, 20, 2);
        attr
    }

    fn encode_attr_ints(name: &str, vals: &[i64]) -> Vec<u8> {
        let mut attr = Vec::new();
        write_bytes_field(&mut attr, 1, name.as_bytes());
        for &v in vals {
            write_varint_field(&mut attr, 8, v as u64);
        }
        // AttributeType::INTS = 7
        write_varint_field(&mut attr, 20, 7);
        attr
    }

    fn encode_gemm_node(
        name: &str,
        a: &str,
        b: &str,
        c: &str,
        out: &str,
        trans_b: i64,
    ) -> Vec<u8> {
        let mut node = Vec::new();
        write_bytes_field(&mut node, 1, a.as_bytes());
        write_bytes_field(&mut node, 1, b.as_bytes());
        write_bytes_field(&mut node, 1, c.as_bytes());
        write_bytes_field(&mut node, 2, out.as_bytes());
        write_bytes_field(&mut node, 3, name.as_bytes());
        write_bytes_field(&mut node, 4, b"Gemm");
        let attr_tb = encode_attr_int("transB", trans_b);
        write_bytes_field(&mut node, 5, &attr_tb);
        node
    }

    fn encode_flatten_node(name: &str, inp: &str, out: &str, axis: i64) -> Vec<u8> {
        let mut node = Vec::new();
        write_bytes_field(&mut node, 1, inp.as_bytes());
        write_bytes_field(&mut node, 2, out.as_bytes());
        write_bytes_field(&mut node, 3, name.as_bytes());
        write_bytes_field(&mut node, 4, b"Flatten");
        let attr_axis = encode_attr_int("axis", axis);
        write_bytes_field(&mut node, 5, &attr_axis);
        node
    }

    fn encode_conv1d_node(
        name: &str,
        inp: &str,
        w: &str,
        b: &str,
        out: &str,
        kernel: i64,
        stride: i64,
        pad: i64,
    ) -> Vec<u8> {
        let mut node = Vec::new();
        write_bytes_field(&mut node, 1, inp.as_bytes());
        write_bytes_field(&mut node, 1, w.as_bytes());
        write_bytes_field(&mut node, 1, b.as_bytes());
        write_bytes_field(&mut node, 2, out.as_bytes());
        write_bytes_field(&mut node, 3, name.as_bytes());
        write_bytes_field(&mut node, 4, b"Conv");
        write_bytes_field(&mut node, 5, &encode_attr_ints("kernel_shape", &[kernel]));
        write_bytes_field(&mut node, 5, &encode_attr_ints("strides", &[stride]));
        write_bytes_field(&mut node, 5, &encode_attr_ints("pads", &[pad, pad]));
        node
    }

    fn encode_unary_node(name: &str, op_type: &str, inp: &str, out: &str) -> Vec<u8> {
        let mut node = Vec::new();
        write_bytes_field(&mut node, 1, inp.as_bytes());
        write_bytes_field(&mut node, 2, out.as_bytes());
        write_bytes_field(&mut node, 3, name.as_bytes());
        write_bytes_field(&mut node, 4, op_type.as_bytes());
        node
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
}
