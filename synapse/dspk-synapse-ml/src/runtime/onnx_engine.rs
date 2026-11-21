//! ONNX graph execution engine powered by `onnx-ir` and [`crate::runtime::burn_engine`].
//!
//! Loads standard `.onnx` model artifacts from disk or memory, pre-uploads constant weights onto
//! the active [`dspk_core::ComputeTarget`], and executes operators with device-resident tensors.

use std::collections::HashMap;
use std::path::Path;

use burn_tensor::activation::{gelu_approximate, leaky_relu, relu, sigmoid, softmax};
use burn_tensor::{Device, Shape, Tensor, TensorData};
use dspk_core::{ComputeTarget, DspError, DspResult};
use onnx_ir::ir::{ArgType, Argument, Node, OnnxGraph, TensorDataExt};
use onnx_ir::node::conv1d::Conv1dNode;
use onnx_ir::node::gemm::GemmNode;
use onnx_ir::node::linear::LinearNode;
use onnx_ir::padding::PaddingConfig1d;
use onnx_ir::OnnxGraphBuilder;

use crate::runtime::burn_engine::{burn_device, PreparedConv1d, PreparedLinear2d};

/// Row-major `[rows, cols]` → `[cols, rows]`.
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Dynamic multi-dimensional float32 tensor passed between ONNX graph nodes.
#[derive(Debug, Clone)]
pub struct RuntimeTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl RuntimeTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> DspResult<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(DspError::InvalidConfig(format!(
                "RuntimeTensor shape {shape:?} expects {expected} elements, got {}",
                data.len()
            )));
        }
        Ok(Self { data, shape })
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

#[derive(Clone)]
enum OnnxValue {
    Dev1(Tensor<1>),
    Dev2(Tensor<2>),
    Dev3(Tensor<3>),
    Host(RuntimeTensor),
}

impl OnnxValue {
    fn from_runtime(rt: RuntimeTensor, dev: &Device) -> Self {
        match rt.shape.as_slice() {
            &[d0] => Self::Dev1(Tensor::<1>::from_data(TensorData::new(rt.data, Shape::new([d0])), dev)),
            &[d0, d1] => Self::Dev2(Tensor::<2>::from_data(TensorData::new(rt.data, Shape::new([d0, d1])), dev)),
            &[d0, d1, d2] => Self::Dev3(Tensor::<3>::from_data(TensorData::new(rt.data, Shape::new([d0, d1, d2])), dev)),
            _ => Self::Host(rt),
        }
    }

    fn to_runtime(&self) -> RuntimeTensor {
        match self {
            Self::Dev1(t) => RuntimeTensor {
                shape: t.dims().to_vec(),
                data: t.clone().into_data().try_to_vec::<f32>().expect("f32 tensor output"),
            },
            Self::Dev2(t) => RuntimeTensor {
                shape: t.dims().to_vec(),
                data: t.clone().into_data().try_to_vec::<f32>().expect("f32 tensor output"),
            },
            Self::Dev3(t) => RuntimeTensor {
                shape: t.dims().to_vec(),
                data: t.clone().into_data().try_to_vec::<f32>().expect("f32 tensor output"),
            },
            Self::Host(rt) => rt.clone(),
        }
    }

    fn into_dev2(self, dev: &Device) -> DspResult<Tensor<2>> {
        match self {
            Self::Dev2(t) => Ok(t),
            other => {
                let rt = other.to_runtime();
                if rt.shape.len() != 2 {
                    return Err(DspError::InvalidConfig(format!("expected 2D tensor, got {:?}", rt.shape)));
                }
                Ok(Tensor::<2>::from_data(TensorData::new(rt.data, Shape::new([rt.shape[0], rt.shape[1]])), dev))
            }
        }
    }

    fn into_dev3(self, dev: &Device) -> DspResult<Tensor<3>> {
        match self {
            Self::Dev3(t) => Ok(t),
            other => {
                let rt = other.to_runtime();
                if rt.shape.len() != 3 {
                    return Err(DspError::InvalidConfig(format!("expected 3D tensor, got {:?}", rt.shape)));
                }
                Ok(Tensor::<3>::from_data(
                    TensorData::new(rt.data, Shape::new([rt.shape[0], rt.shape[1], rt.shape[2]])),
                    dev,
                ))
            }
        }
    }
}

#[derive(Clone, Debug)]
enum PreparedNodeOp {
    Linear {
        linear: PreparedLinear2d,
        in_dim: usize,
        out_dim: usize,
    },
    Gemm {
        linear: PreparedLinear2d,
        trans_a: bool,
        alpha: f32,
        beta: f32,
        scaled_bias: Option<Tensor<2>>,
        k: usize,
        out_dim: usize,
    },
    Conv1d {
        conv: PreparedConv1d,
        in_ch: usize,
    },
}

fn extract_constant(arg: &Argument) -> DspResult<Option<RuntimeTensor>> {
    let Some(data) = arg.value() else {
        return Ok(None);
    };
    let shape: Vec<usize> = data.shape().iter().copied().collect();
    let floats = data
        .to_f32_vec()
        .map_err(|e| DspError::InvalidConfig(format!("ONNX constant f32 conversion failed: {e:?}")))?;
    Ok(Some(RuntimeTensor::new(floats, shape)?))
}

/// Self-contained ONNX model runner backed by `onnx-ir` and [`crate::runtime::burn_engine`].
pub struct OnnxRuntimeSession {
    graph: OnnxGraph,
    target: ComputeTarget,
    device: Device,
    constants: HashMap<String, RuntimeTensor>,
    prepared_ops: Vec<Option<PreparedNodeOp>>,
}

impl OnnxRuntimeSession {
    fn from_graph(graph: OnnxGraph, target: ComputeTarget) -> DspResult<Self> {
        let device = burn_device(target)?;
        let mut constants: HashMap<String, RuntimeTensor> = HashMap::new();
        for node in &graph.nodes {
            for arg in node.inputs() {
                if !arg.name.is_empty() && !constants.contains_key(&arg.name) {
                    if let Some(rt) = extract_constant(arg)? {
                        constants.insert(arg.name.clone(), rt);
                    }
                }
            }
        }
        let mut prepared_ops = Vec::with_capacity(graph.nodes.len());
        for node in &graph.nodes {
            let prep = Self::prepare_node_op(node, &device)?;
            prepared_ops.push(prep);
        }
        Ok(Self { graph, target, device, constants, prepared_ops })
    }

    fn prepare_node_op(node: &Node, dev: &Device) -> DspResult<Option<PreparedNodeOp>> {
        match node {
            Node::Linear(n) => {
                let Some(w) = extract_constant(&n.inputs[1])? else { return Ok(None) };
                if w.shape.len() != 2 {
                    return Ok(None);
                }
                let b = if n.inputs.len() > 2 && !n.inputs[2].is_optional() {
                    match extract_constant(&n.inputs[2])? {
                        Some(bt) => Some(bt),
                        None => return Ok(None),
                    }
                } else {
                    None
                };
                if n.config.transpose_weight {
                    let (in_dim, out_dim) = (w.shape[0], w.shape[1]);
                    let linear = PreparedLinear2d::from_pretransposed(dev, &w.data, in_dim, out_dim, b.as_ref().map(|t| t.data.as_slice()))?;
                    Ok(Some(PreparedNodeOp::Linear { linear, in_dim, out_dim }))
                } else {
                    let (out_dim, in_dim) = (w.shape[0], w.shape[1]);
                    let linear = PreparedLinear2d::new(dev, &w.data, out_dim, in_dim, b.as_ref().map(|t| t.data.as_slice()))?;
                    Ok(Some(PreparedNodeOp::Linear { linear, in_dim, out_dim }))
                }
            }
            Node::MatMul(n) => {
                let Some(w) = extract_constant(&n.inputs[1])? else { return Ok(None) };
                if w.shape.len() != 2 {
                    return Ok(None);
                }
                let (in_dim, out_dim) = (w.shape[0], w.shape[1]);
                let linear = PreparedLinear2d::from_pretransposed(dev, &w.data, in_dim, out_dim, None)?;
                Ok(Some(PreparedNodeOp::Linear { linear, in_dim, out_dim }))
            }
            Node::Gemm(n) => {
                let Some(b) = extract_constant(&n.inputs[1])? else { return Ok(None) };
                if b.shape.len() != 2 {
                    return Ok(None);
                }
                let c = if n.inputs.len() > 2 && !n.inputs[2].is_optional() {
                    match extract_constant(&n.inputs[2])? {
                        Some(ct) => Some(ct),
                        None => return Ok(None),
                    }
                } else {
                    None
                };
                let (k, out_dim, linear) = if n.config.trans_b != 0 {
                    let (out_d, k_b) = (b.shape[0], b.shape[1]);
                    (k_b, out_d, PreparedLinear2d::new(dev, &b.data, out_d, k_b, None)?)
                } else {
                    let (k_b, out_d) = (b.shape[0], b.shape[1]);
                    (k_b, out_d, PreparedLinear2d::from_pretransposed(dev, &b.data, k_b, out_d, None)?)
                };
                let scaled_bias = if let Some(bias) = c {
                    if n.config.beta != 0.0 && bias.data.len() == out_dim {
                        let scaled: Vec<f32> = bias.data.iter().map(|&v| v * n.config.beta).collect();
                        Some(Tensor::<2>::from_data(TensorData::new(scaled, Shape::new([1, out_dim])), dev))
                    } else if n.config.beta != 0.0 && bias.data.len() == 1 {
                        Some(Tensor::<2>::from_data(
                            TensorData::new(vec![bias.data[0] * n.config.beta; out_dim], Shape::new([1, out_dim])),
                            dev,
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                };
                Ok(Some(PreparedNodeOp::Gemm {
                    linear,
                    trans_a: n.config.trans_a != 0,
                    alpha: n.config.alpha,
                    beta: n.config.beta,
                    scaled_bias,
                    k,
                    out_dim,
                }))
            }
            Node::Conv1d(n) => {
                let Some(w) = extract_constant(&n.inputs[1])? else { return Ok(None) };
                if w.shape.len() != 3 {
                    return Ok(None);
                }
                let b = if n.inputs.len() > 2 && !n.inputs[2].is_optional() {
                    match extract_constant(&n.inputs[2])? {
                        Some(bt) => Some(bt),
                        None => return Ok(None),
                    }
                } else {
                    None
                };
                let (out_ch, in_per_group, kernel_size) = (w.shape[0], w.shape[1], w.shape[2]);
                let padding = match &n.config.padding {
                    PaddingConfig1d::Valid => 0,
                    PaddingConfig1d::Explicit(l, _r) => *l,
                };
                let stride = n.config.stride.max(1);
                let dilation = n.config.dilation.max(1);
                let groups = n.config.groups.max(1);
                let conv = PreparedConv1d::new(
                    dev,
                    &w.data,
                    out_ch,
                    in_per_group,
                    kernel_size,
                    b.as_ref().map(|t| t.data.as_slice()),
                    stride,
                    padding,
                    dilation,
                    groups,
                )?;
                Ok(Some(PreparedNodeOp::Conv1d {
                    conv,
                    in_ch: in_per_group * groups,
                }))
            }
            _ => Ok(None),
        }
    }

    /// Parses an `.onnx` model file from disk and binds it to `target`.
    pub fn from_file(path: impl AsRef<Path>, target: ComputeTarget) -> DspResult<Self> {
        let path_ref = path.as_ref();
        let graph = OnnxGraphBuilder::new()
            .parse_file(path_ref)
            .map_err(|e| DspError::InvalidConfig(format!("ONNX parse failed for {:?}: {e:?}", path_ref)))?;
        Self::from_graph(graph, target)
    }

    /// Parses an `.onnx` model from an in-memory byte slice and binds it to `target`.
    pub fn from_bytes(bytes: &[u8], target: ComputeTarget) -> DspResult<Self> {
        let graph = OnnxGraphBuilder::new()
            .parse_bytes(bytes)
            .map_err(|e| DspError::InvalidConfig(format!("ONNX parse_bytes failed: {e:?}")))?;
        Self::from_graph(graph, target)
    }

    pub fn target(&self) -> ComputeTarget {
        self.target
    }

    pub fn node_count(&self) -> usize {
        self.graph.nodes.len()
    }

    pub fn input_names(&self) -> Vec<String> {
        self.graph.inputs.iter().map(|i| i.name.clone()).collect()
    }

    pub fn output_names(&self) -> Vec<String> {
        self.graph.outputs.iter().map(|o| o.name.clone()).collect()
    }

    /// Executes the ONNX graph on a single input tensor and returns the primary output tensor.
    pub fn run(&self, input: RuntimeTensor) -> DspResult<RuntimeTensor> {
        let first_input = self
            .graph
            .inputs
            .first()
            .ok_or_else(|| DspError::InvalidConfig("ONNX graph has no inputs".to_string()))?
            .name
            .clone();

        let mut inputs = HashMap::new();
        inputs.insert(first_input, input);
        let mut outputs = self.run_named(inputs)?;

        let first_output = self
            .graph
            .outputs
            .first()
            .ok_or_else(|| DspError::InvalidConfig("ONNX graph has no outputs".to_string()))?
            .name
            .clone();

        outputs
            .remove(&first_output)
            .ok_or_else(|| DspError::InvalidConfig(format!("ONNX output '{first_output}' not produced")))
    }

    /// Executes the ONNX graph in topological order on named input tensors.
    pub fn run_named(
        &self,
        env: HashMap<String, RuntimeTensor>,
    ) -> DspResult<HashMap<String, RuntimeTensor>> {
        let mut dev_env: HashMap<String, OnnxValue> = env
            .into_iter()
            .map(|(k, v)| (k, OnnxValue::from_runtime(v, &self.device)))
            .collect();

        for (idx, node) in self.graph.nodes.iter().enumerate() {
            let prep = self.prepared_ops.get(idx).and_then(|p| p.as_ref());
            self.execute_node(node, prep, &mut dev_env)?;
        }

        let mut out = HashMap::new();
        for spec in &self.graph.outputs {
            if let Some(v) = dev_env.remove(&spec.name) {
                out.insert(spec.name.clone(), v.to_runtime());
            }
        }
        Ok(out)
    }

    fn resolve_value(&self, arg: &Argument, env: &HashMap<String, OnnxValue>) -> DspResult<OnnxValue> {
        if let Some(c) = self.constants.get(&arg.name) {
            return Ok(OnnxValue::Host(c.clone()));
        }
        if let Some(c) = extract_constant(arg)? {
            return Ok(OnnxValue::Host(c));
        }
        if let ArgType::ScalarNative(_) | ArgType::ScalarTensor(_) = &arg.ty {
            if let Some(t) = env.get(&arg.name) {
                return Ok(t.clone());
            }
        }
        env.get(&arg.name).cloned().ok_or_else(|| {
            DspError::InvalidConfig(format!("ONNX tensor '{}' not found in execution scope", arg.name))
        })
    }

    fn execute_node(
        &self,
        node: &Node,
        prep: Option<&PreparedNodeOp>,
        env: &mut HashMap<String, OnnxValue>,
    ) -> DspResult<()> {
        match node {
            Node::Relu(n) => {
                let x = self.resolve_value(&n.inputs[0], env)?;
                let out = match x {
                    OnnxValue::Dev1(t) => OnnxValue::Dev1(relu(t)),
                    OnnxValue::Dev2(t) => OnnxValue::Dev2(relu(t)),
                    OnnxValue::Dev3(t) => OnnxValue::Dev3(relu(t)),
                    OnnxValue::Host(rt) => {
                        let data = rt.data.iter().map(|&v| v.max(0.0)).collect();
                        OnnxValue::Host(RuntimeTensor { data, shape: rt.shape })
                    }
                };
                env.insert(n.outputs[0].name.clone(), out);
            }
            Node::Sigmoid(n) => {
                let x = self.resolve_value(&n.inputs[0], env)?;
                let out = match x {
                    OnnxValue::Dev1(t) => OnnxValue::Dev1(sigmoid(t)),
                    OnnxValue::Dev2(t) => OnnxValue::Dev2(sigmoid(t)),
                    OnnxValue::Dev3(t) => OnnxValue::Dev3(sigmoid(t)),
                    OnnxValue::Host(rt) => {
                        let data = rt.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
                        OnnxValue::Host(RuntimeTensor { data, shape: rt.shape })
                    }
                };
                env.insert(n.outputs[0].name.clone(), out);
            }
            Node::Tanh(n) => {
                let x = self.resolve_value(&n.inputs[0], env)?;
                let out = match x {
                    OnnxValue::Dev1(t) => OnnxValue::Dev1(t.tanh()),
                    OnnxValue::Dev2(t) => OnnxValue::Dev2(t.tanh()),
                    OnnxValue::Dev3(t) => OnnxValue::Dev3(t.tanh()),
                    OnnxValue::Host(rt) => {
                        let data = rt.data.iter().map(|&v| v.tanh()).collect();
                        OnnxValue::Host(RuntimeTensor { data, shape: rt.shape })
                    }
                };
                env.insert(n.outputs[0].name.clone(), out);
            }
            Node::Gelu(n) => {
                let x = self.resolve_value(&n.inputs[0], env)?;
                let out = match x {
                    OnnxValue::Dev1(t) => OnnxValue::Dev1(gelu_approximate(t)),
                    OnnxValue::Dev2(t) => OnnxValue::Dev2(gelu_approximate(t)),
                    OnnxValue::Dev3(t) => OnnxValue::Dev3(gelu_approximate(t)),
                    OnnxValue::Host(rt) => {
                        let k = (2.0f32 / std::f32::consts::PI).sqrt();
                        let data = rt
                            .data
                            .iter()
                            .map(|&v| 0.5 * v * (1.0 + (k * (v + 0.044715 * v * v * v)).tanh()))
                            .collect();
                        OnnxValue::Host(RuntimeTensor { data, shape: rt.shape })
                    }
                };
                env.insert(n.outputs[0].name.clone(), out);
            }
            Node::LeakyRelu(n) => {
                let x = self.resolve_value(&n.inputs[0], env)?;
                let out = match x {
                    OnnxValue::Dev1(t) => OnnxValue::Dev1(leaky_relu(t, n.config.alpha)),
                    OnnxValue::Dev2(t) => OnnxValue::Dev2(leaky_relu(t, n.config.alpha)),
                    OnnxValue::Dev3(t) => OnnxValue::Dev3(leaky_relu(t, n.config.alpha)),
                    OnnxValue::Host(rt) => {
                        let alpha = n.config.alpha as f32;
                        let data = rt.data.iter().map(|&v| if v >= 0.0 { v } else { alpha * v }).collect();
                        OnnxValue::Host(RuntimeTensor { data, shape: rt.shape })
                    }
                };
                env.insert(n.outputs[0].name.clone(), out);
            }
            Node::Softmax(n) => {
                let x = self.resolve_value(&n.inputs[0], env)?;
                let out = match x {
                    OnnxValue::Dev1(t) => OnnxValue::Dev1(softmax(t, 0)),
                    OnnxValue::Dev2(t) => OnnxValue::Dev2(softmax(t, 1)),
                    OnnxValue::Dev3(t) => OnnxValue::Dev3(softmax(t, 2)),
                    OnnxValue::Host(rt) => {
                        let last_dim = *rt.shape.last().unwrap_or(&1).max(&1);
                        let mut data = rt.data;
                        for row in data.chunks_exact_mut(last_dim) {
                            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let mut sum = 0.0f32;
                            for v in row.iter_mut() {
                                *v = (*v - max_v).exp();
                                sum += *v;
                            }
                            let inv = 1.0 / sum.max(1e-12);
                            for v in row.iter_mut() {
                                *v *= inv;
                            }
                        }
                        OnnxValue::Host(RuntimeTensor { data, shape: rt.shape })
                    }
                };
                env.insert(n.outputs[0].name.clone(), out);
            }
            Node::Add(n) => self.exec_binary(&n.inputs, &n.outputs[0].name, env, BinaryOp::Add)?,
            Node::Sub(n) => self.exec_binary(&n.inputs, &n.outputs[0].name, env, BinaryOp::Sub)?,
            Node::Mul(n) => self.exec_binary(&n.inputs, &n.outputs[0].name, env, BinaryOp::Mul)?,
            Node::Div(n) => self.exec_binary(&n.inputs, &n.outputs[0].name, env, BinaryOp::Div)?,
            Node::Flatten(n) => {
                let x = self.resolve_value(&n.inputs[0], env)?;
                let out = match x {
                    OnnxValue::Dev1(t) => {
                        let d0 = t.dims()[0];
                        let [dim0, dim1] = if n.config.axis == 0 { [1, d0] } else { [d0, 1] };
                        OnnxValue::Dev2(t.reshape([dim0, dim1]))
                    }
                    OnnxValue::Dev2(t) => {
                        let [d0, d1] = t.dims();
                        let dims = match n.config.axis {
                            0 => [1, d0 * d1],
                            1 => [d0, d1],
                            _ => [d0 * d1, 1],
                        };
                        OnnxValue::Dev2(t.reshape(dims))
                    }
                    OnnxValue::Dev3(t) => {
                        let [d0, d1, d2] = t.dims();
                        let dims = match n.config.axis {
                            0 => [1, d0 * d1 * d2],
                            1 => [d0, d1 * d2],
                            2 => [d0 * d1, d2],
                            _ => [d0 * d1 * d2, 1],
                        };
                        OnnxValue::Dev2(t.reshape(dims))
                    }
                    OnnxValue::Host(rt) => {
                        let rank = rt.shape.len();
                        let axis = n.config.axis.min(rank);
                        let dim0: usize = if axis == 0 { 1 } else { rt.shape[..axis].iter().product() };
                        let dim1: usize = rt.shape[axis..].iter().product();
                        OnnxValue::Dev2(Tensor::<2>::from_data(
                            TensorData::new(rt.data, Shape::new([dim0, dim1])),
                            &self.device,
                        ))
                    }
                };
                env.insert(n.outputs[0].name.clone(), out);
            }
            Node::Identity(n) => {
                let x = self.resolve_value(&n.inputs[0], env)?;
                env.insert(n.outputs[0].name.clone(), x);
            }
            Node::Gemm(n) => self.exec_gemm(n, prep, env)?,
            Node::Linear(n) => self.exec_linear(n, prep, env)?,
            Node::MatMul(n) => {
                let a = self.resolve_value(&n.inputs[0], env)?.into_dev2(&self.device)?;
                if let Some(PreparedNodeOp::Linear { linear, in_dim, .. }) = prep {
                    let [_batch, k] = a.dims();
                    if k != *in_dim {
                        return Err(DspError::InvalidConfig(format!(
                            "ONNX MatMul incompatible shapes [{_batch}, {k}] and [{in_dim}, ...]"
                        )));
                    }
                    env.insert(n.outputs[0].name.clone(), OnnxValue::Dev2(linear.forward(a)));
                } else {
                    let b = self.resolve_value(&n.inputs[1], env)?.into_dev2(&self.device)?;
                    if a.dims()[1] != b.dims()[0] {
                        return Err(DspError::InvalidConfig(format!(
                            "ONNX MatMul incompatible shapes {:?} and {:?}",
                            a.dims(),
                            b.dims()
                        )));
                    }
                    env.insert(n.outputs[0].name.clone(), OnnxValue::Dev2(a.matmul(b)));
                }
            }
            Node::Conv1d(n) => self.exec_conv1d(n, prep, env)?,
            Node::GlobalAveragePool(n) => {
                let x = self.resolve_value(&n.inputs[0], env)?;
                let out = match x {
                    OnnxValue::Dev3(t) => OnnxValue::Dev3(t.mean_dim(2)),
                    other => {
                        let t = other.into_dev3(&self.device)?;
                        OnnxValue::Dev3(t.mean_dim(2))
                    }
                };
                env.insert(n.outputs[0].name.clone(), out);
            }
            other => {
                return Err(DspError::InvalidConfig(format!(
                    "Unsupported ONNX operator node: {:?}",
                    other.name()
                )));
            }
        }
        Ok(())
    }

    fn exec_binary(
        &self,
        inputs: &[Argument],
        out_name: &str,
        env: &mut HashMap<String, OnnxValue>,
        op: BinaryOp,
    ) -> DspResult<()> {
        let a = self.resolve_value(&inputs[0], env)?;
        let b = self.resolve_value(&inputs[1], env)?;
        if let Some(dev_out) = try_binary_dev(&a, &b, op) {
            env.insert(out_name.to_string(), dev_out);
            return Ok(());
        }
        let a_rt = a.to_runtime();
        let b_rt = b.to_runtime();
        let host_op: fn(f32, f32) -> f32 = match op {
            BinaryOp::Add => |x, y| x + y,
            BinaryOp::Sub => |x, y| x - y,
            BinaryOp::Mul => |x, y| x * y,
            BinaryOp::Div => |x, y| x / y,
        };
        let data = if a_rt.data.len() == b_rt.data.len() {
            a_rt.data.iter().zip(b_rt.data.iter()).map(|(&x, &y)| host_op(x, y)).collect()
        } else if b_rt.data.len() == 1 {
            let s = b_rt.data[0];
            a_rt.data.iter().map(|&x| host_op(x, s)).collect()
        } else if a_rt.data.len() == 1 {
            let s = a_rt.data[0];
            b_rt.data.iter().map(|&y| host_op(s, y)).collect()
        } else if a_rt.data.len() % b_rt.data.len() == 0 {
            let stride = b_rt.data.len();
            a_rt.data
                .iter()
                .enumerate()
                .map(|(i, &x)| host_op(x, b_rt.data[i % stride]))
                .collect()
        } else {
            return Err(DspError::InvalidConfig(format!(
                "ONNX binary broadcast unsupported for {:?} and {:?}",
                a_rt.shape, b_rt.shape
            )));
        };
        let shape = if a_rt.numel() >= b_rt.numel() { a_rt.shape } else { b_rt.shape };
        env.insert(out_name.to_string(), OnnxValue::from_runtime(RuntimeTensor { data, shape }, &self.device));
        Ok(())
    }

    fn exec_gemm(
        &self,
        n: &GemmNode,
        prep: Option<&PreparedNodeOp>,
        env: &mut HashMap<String, OnnxValue>,
    ) -> DspResult<()> {
        if let Some(PreparedNodeOp::Gemm { linear, trans_a, alpha, beta, scaled_bias, k, out_dim }) = prep {
            let mut a = self.resolve_value(&n.inputs[0], env)?.into_dev2(&self.device)?;
            if *trans_a {
                a = a.transpose();
            }
            if a.dims()[1] != *k {
                return Err(DspError::InvalidConfig(format!(
                    "ONNX Gemm inner dimension mismatch: {} vs {k}",
                    a.dims()[1]
                )));
            }
            let mut y = linear.forward(a);
            if (*alpha - 1.0).abs() > 1e-6 {
                y = y.mul_scalar(*alpha);
            }
            if let Some(bias) = scaled_bias {
                y = y + bias.clone();
            } else if *beta != 0.0 && n.inputs.len() > 2 && !n.inputs[2].is_optional() {
                let c_rt = self.resolve_value(&n.inputs[2], env)?.to_runtime();
                if c_rt.data.len() == *out_dim {
                    let scaled: Vec<f32> = c_rt.data.iter().map(|&v| v * *beta).collect();
                    let b_t = Tensor::<2>::from_data(TensorData::new(scaled, Shape::new([1, *out_dim])), &self.device);
                    y = y + b_t;
                }
            }
            env.insert(n.outputs[0].name.clone(), OnnxValue::Dev2(y));
            return Ok(());
        }

        let a = self.resolve_value(&n.inputs[0], env)?.to_runtime();
        let b = self.resolve_value(&n.inputs[1], env)?.to_runtime();
        let c = if n.inputs.len() > 2 && !n.inputs[2].is_optional() {
            Some(self.resolve_value(&n.inputs[2], env)?.to_runtime())
        } else {
            None
        };

        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(DspError::InvalidConfig(format!(
                "ONNX Gemm requires 2D tensors, got {:?} and {:?}",
                a.shape, b.shape
            )));
        }

        let (m, k_a) = if n.config.trans_a == 0 {
            (a.shape[0], a.shape[1])
        } else {
            (a.shape[1], a.shape[0])
        };
        let a_row = if n.config.trans_a == 0 {
            a.data
        } else {
            transpose_2d(&a.data, a.shape[0], a.shape[1])
        };
        let (out_dim, k_b, w_out_in) = if n.config.trans_b != 0 {
            (b.shape[0], b.shape[1], b.data)
        } else {
            (b.shape[1], b.shape[0], transpose_2d(&b.data, b.shape[0], b.shape[1]))
        };
        if k_a != k_b {
            return Err(DspError::InvalidConfig(format!("ONNX Gemm inner dimension mismatch: {k_a} vs {k_b}")));
        }

        let a_t = Tensor::<2>::from_data(TensorData::new(a_row, Shape::new([m, k_a])), &self.device);
        let w_t = Tensor::<2>::from_data(TensorData::new(w_out_in, Shape::new([out_dim, k_a])), &self.device).transpose();
        let mut y = a_t.matmul(w_t);
        if (n.config.alpha - 1.0).abs() > 1e-6 {
            y = y.mul_scalar(n.config.alpha);
        }
        if let Some(bias) = c {
            if n.config.beta != 0.0 && bias.data.len() == out_dim {
                let scaled: Vec<f32> = bias.data.iter().map(|&v| v * n.config.beta).collect();
                let b_t = Tensor::<2>::from_data(TensorData::new(scaled, Shape::new([1, out_dim])), &self.device);
                y = y + b_t;
            }
        }
        env.insert(n.outputs[0].name.clone(), OnnxValue::Dev2(y));
        Ok(())
    }

    fn exec_linear(
        &self,
        n: &LinearNode,
        prep: Option<&PreparedNodeOp>,
        env: &mut HashMap<String, OnnxValue>,
    ) -> DspResult<()> {
        let x = self.resolve_value(&n.inputs[0], env)?;
        if let Some(PreparedNodeOp::Linear { linear, in_dim, out_dim }) = prep {
            let (orig_shape, batch, x_2d) = match x {
                OnnxValue::Dev2(t) => {
                    let [b, d] = t.dims();
                    (vec![b, d], b, t)
                }
                other => {
                    let rt = other.to_runtime();
                    let d = *rt.shape.last().unwrap_or(&1);
                    let batch = rt.numel() / d.max(1);
                    let t = Tensor::<2>::from_data(TensorData::new(rt.data, Shape::new([batch, d])), &self.device);
                    (rt.shape, batch, t)
                }
            };
            if x_2d.dims()[1] != *in_dim {
                return Err(DspError::InvalidConfig(format!(
                    "ONNX Linear expects in_features={in_dim}, got {}",
                    x_2d.dims()[1]
                )));
            }
            let y = linear.forward(x_2d);
            let mut out_shape = orig_shape[..orig_shape.len() - 1].to_vec();
            out_shape.push(*out_dim);
            if out_shape.len() == 2 {
                env.insert(n.outputs[0].name.clone(), OnnxValue::Dev2(y));
            } else {
                let mut rt = OnnxValue::Dev2(y).to_runtime();
                rt.shape = out_shape;
                let _ = batch;
                env.insert(n.outputs[0].name.clone(), OnnxValue::from_runtime(rt, &self.device));
            }
            return Ok(());
        }

        let x_rt = x.to_runtime();
        let w = self.resolve_value(&n.inputs[1], env)?.to_runtime();
        let b = if n.inputs.len() > 2 && !n.inputs[2].is_optional() {
            Some(self.resolve_value(&n.inputs[2], env)?.to_runtime())
        } else {
            None
        };
        let in_dim = *x_rt.shape.last().unwrap_or(&1);
        let batch = x_rt.numel() / in_dim.max(1);
        let (out_dim, linear) = if n.config.transpose_weight {
            (
                w.shape[1],
                PreparedLinear2d::from_pretransposed(&self.device, &w.data, in_dim, w.shape[1], b.as_ref().map(|t| t.data.as_slice()))?,
            )
        } else {
            (
                w.shape[0],
                PreparedLinear2d::new(&self.device, &w.data, w.shape[0], in_dim, b.as_ref().map(|t| t.data.as_slice()))?,
            )
        };
        let x_t = Tensor::<2>::from_data(TensorData::new(x_rt.data, Shape::new([batch, in_dim])), &self.device);
        let y = linear.forward(x_t);
        let mut out_shape = x_rt.shape[..x_rt.shape.len() - 1].to_vec();
        out_shape.push(out_dim);
        if out_shape.len() == 2 {
            env.insert(n.outputs[0].name.clone(), OnnxValue::Dev2(y));
        } else {
            let mut rt = OnnxValue::Dev2(y).to_runtime();
            rt.shape = out_shape;
            env.insert(n.outputs[0].name.clone(), OnnxValue::from_runtime(rt, &self.device));
        }
        Ok(())
    }

    fn exec_conv1d(
        &self,
        n: &Conv1dNode,
        prep: Option<&PreparedNodeOp>,
        env: &mut HashMap<String, OnnxValue>,
    ) -> DspResult<()> {
        let x = self.resolve_value(&n.inputs[0], env)?.into_dev3(&self.device)?;
        if let Some(PreparedNodeOp::Conv1d { conv, in_ch }) = prep {
            if x.dims()[1] != *in_ch {
                return Err(DspError::InvalidConfig(format!(
                    "ONNX Conv1d expects in_channels={in_ch}, got {}",
                    x.dims()[1]
                )));
            }
            env.insert(n.outputs[0].name.clone(), OnnxValue::Dev3(conv.forward(x)));
            return Ok(());
        }

        let w = self.resolve_value(&n.inputs[1], env)?.to_runtime();
        let b = if n.inputs.len() > 2 && !n.inputs[2].is_optional() {
            Some(self.resolve_value(&n.inputs[2], env)?.to_runtime())
        } else {
            None
        };
        if w.shape.len() != 3 {
            return Err(DspError::InvalidConfig(format!(
                "ONNX Conv1d expects 3D weight, got {:?}",
                w.shape
            )));
        }
        let padding = match &n.config.padding {
            PaddingConfig1d::Valid => 0,
            PaddingConfig1d::Explicit(l, _r) => *l,
        };
        let conv = PreparedConv1d::new(
            &self.device,
            &w.data,
            w.shape[0],
            w.shape[1],
            w.shape[2],
            b.as_ref().map(|t| t.data.as_slice()),
            n.config.stride.max(1),
            padding,
            n.config.dilation.max(1),
            n.config.groups.max(1),
        )?;
        env.insert(n.outputs[0].name.clone(), OnnxValue::Dev3(conv.forward(x)));
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

fn broadcastable<const D: usize>(a: [usize; D], b: [usize; D]) -> bool {
    (0..D).all(|i| a[i] == b[i] || a[i] == 1 || b[i] == 1)
}

fn try_binary_dev(a: &OnnxValue, b: &OnnxValue, op: BinaryOp) -> Option<OnnxValue> {
    match (a, b) {
        (OnnxValue::Dev1(ta), OnnxValue::Dev1(tb)) if broadcastable(ta.dims(), tb.dims()) => Some(OnnxValue::Dev1(match op {
            BinaryOp::Add => ta.clone() + tb.clone(),
            BinaryOp::Sub => ta.clone() - tb.clone(),
            BinaryOp::Mul => ta.clone() * tb.clone(),
            BinaryOp::Div => ta.clone() / tb.clone(),
        })),
        (OnnxValue::Dev2(ta), OnnxValue::Dev2(tb)) if broadcastable(ta.dims(), tb.dims()) => Some(OnnxValue::Dev2(match op {
            BinaryOp::Add => ta.clone() + tb.clone(),
            BinaryOp::Sub => ta.clone() - tb.clone(),
            BinaryOp::Mul => ta.clone() * tb.clone(),
            BinaryOp::Div => ta.clone() / tb.clone(),
        })),
        (OnnxValue::Dev3(ta), OnnxValue::Dev3(tb)) if broadcastable(ta.dims(), tb.dims()) => Some(OnnxValue::Dev3(match op {
            BinaryOp::Add => ta.clone() + tb.clone(),
            BinaryOp::Sub => ta.clone() - tb.clone(),
            BinaryOp::Mul => ta.clone() * tb.clone(),
            BinaryOp::Div => ta.clone() / tb.clone(),
        })),
        (OnnxValue::Dev1(ta), OnnxValue::Host(rb)) if rb.data.len() == 1 => {
            let s = rb.data[0];
            Some(OnnxValue::Dev1(match op {
                BinaryOp::Add => ta.clone().add_scalar(s),
                BinaryOp::Sub => ta.clone().sub_scalar(s),
                BinaryOp::Mul => ta.clone().mul_scalar(s),
                BinaryOp::Div => ta.clone().div_scalar(s),
            }))
        }
        (OnnxValue::Dev2(ta), OnnxValue::Host(rb)) if rb.data.len() == 1 => {
            let s = rb.data[0];
            Some(OnnxValue::Dev2(match op {
                BinaryOp::Add => ta.clone().add_scalar(s),
                BinaryOp::Sub => ta.clone().sub_scalar(s),
                BinaryOp::Mul => ta.clone().mul_scalar(s),
                BinaryOp::Div => ta.clone().div_scalar(s),
            }))
        }
        (OnnxValue::Dev3(ta), OnnxValue::Host(rb)) if rb.data.len() == 1 => {
            let s = rb.data[0];
            Some(OnnxValue::Dev3(match op {
                BinaryOp::Add => ta.clone().add_scalar(s),
                BinaryOp::Sub => ta.clone().sub_scalar(s),
                BinaryOp::Mul => ta.clone().mul_scalar(s),
                BinaryOp::Div => ta.clone().div_scalar(s),
            }))
        }
        _ => None,
    }
}
