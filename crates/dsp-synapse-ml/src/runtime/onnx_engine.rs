//! ONNX graph execution engine powered by `onnx-ir` and [`crate::runtime::burn_engine`].
//!
//! Loads standard `.onnx` model artifacts from disk or memory and executes their
//! operator graphs on the active [`dsp_core::ComputeTarget`].

use std::collections::HashMap;
use std::path::Path;

use dsp_core::{ComputeTarget, DspError, DspResult};
use onnx_ir::ir::{ArgType, Node, OnnxGraph, TensorDataExt};
use onnx_ir::node::conv1d::Conv1dNode;
use onnx_ir::node::gemm::GemmNode;
use onnx_ir::node::linear::LinearNode;
use onnx_ir::padding::PaddingConfig1d;
use onnx_ir::OnnxGraphBuilder;

use crate::runtime::burn_engine::{burn_conv1d, burn_linear_2d};

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

/// Self-contained ONNX model runner backed by `onnx-ir` and [`crate::runtime::burn_engine`].
pub struct OnnxRuntimeSession {
    graph: OnnxGraph,
    target: ComputeTarget,
}

impl OnnxRuntimeSession {
    /// Parses an `.onnx` model file from disk and binds it to `target`.
    pub fn from_file(path: impl AsRef<Path>, target: ComputeTarget) -> DspResult<Self> {
        let path_ref = path.as_ref();
        let graph = OnnxGraphBuilder::new()
            .parse_file(path_ref)
            .map_err(|e| DspError::InvalidConfig(format!("ONNX parse failed for {:?}: {e:?}", path_ref)))?;
        Ok(Self { graph, target })
    }

    /// Parses an `.onnx` model from an in-memory byte slice and binds it to `target`.
    pub fn from_bytes(bytes: &[u8], target: ComputeTarget) -> DspResult<Self> {
        let graph = OnnxGraphBuilder::new()
            .parse_bytes(bytes)
            .map_err(|e| DspError::InvalidConfig(format!("ONNX parse_bytes failed: {e:?}")))?;
        Ok(Self { graph, target })
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
        mut env: HashMap<String, RuntimeTensor>,
    ) -> DspResult<HashMap<String, RuntimeTensor>> {
        for node in &self.graph.nodes {
            self.execute_node(node, &mut env)?;
        }

        let mut out = HashMap::new();
        for spec in &self.graph.outputs {
            if let Some(t) = env.remove(&spec.name) {
                out.insert(spec.name.clone(), t);
            }
        }
        Ok(out)
    }

    fn resolve_arg(
        &self,
        arg: &onnx_ir::Argument,
        env: &HashMap<String, RuntimeTensor>,
    ) -> DspResult<RuntimeTensor> {
        if let Some(data) = arg.value() {
            let shape: Vec<usize> = data.shape.iter().copied().collect();
            let floats = data
                .to_f32_vec()
                .map_err(|e| DspError::InvalidConfig(format!("ONNX constant f32 conversion failed: {e:?}")))?;
            return RuntimeTensor::new(floats, shape);
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
        env: &mut HashMap<String, RuntimeTensor>,
    ) -> DspResult<()> {
        match node {
            Node::Relu(n) => {
                let x = self.resolve_arg(&n.inputs[0], env)?;
                let data = x.data.iter().map(|&v| v.max(0.0)).collect();
                env.insert(n.outputs[0].name.clone(), RuntimeTensor { data, shape: x.shape });
            }
            Node::Sigmoid(n) => {
                let x = self.resolve_arg(&n.inputs[0], env)?;
                let data = x.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
                env.insert(n.outputs[0].name.clone(), RuntimeTensor { data, shape: x.shape });
            }
            Node::Tanh(n) => {
                let x = self.resolve_arg(&n.inputs[0], env)?;
                let data = x.data.iter().map(|&v| v.tanh()).collect();
                env.insert(n.outputs[0].name.clone(), RuntimeTensor { data, shape: x.shape });
            }
            Node::Gelu(n) => {
                let x = self.resolve_arg(&n.inputs[0], env)?;
                let k = (2.0f32 / std::f32::consts::PI).sqrt();
                let data = x
                    .data
                    .iter()
                    .map(|&v| 0.5 * v * (1.0 + (k * (v + 0.044715 * v * v * v)).tanh()))
                    .collect();
                env.insert(n.outputs[0].name.clone(), RuntimeTensor { data, shape: x.shape });
            }
            Node::LeakyRelu(n) => {
                let x = self.resolve_arg(&n.inputs[0], env)?;
                let alpha = n.config.alpha as f32;
                let data = x
                    .data
                    .iter()
                    .map(|&v| if v >= 0.0 { v } else { alpha * v })
                    .collect();
                env.insert(n.outputs[0].name.clone(), RuntimeTensor { data, shape: x.shape });
            }
            Node::Softmax(n) => {
                let x = self.resolve_arg(&n.inputs[0], env)?;
                let last_dim = *x.shape.last().unwrap_or(&1).max(&1);
                let mut data = x.data.clone();
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
                env.insert(n.outputs[0].name.clone(), RuntimeTensor { data, shape: x.shape });
            }
            Node::Add(n) => self.exec_binary(&n.inputs, &n.outputs[0].name, env, |a, b| a + b)?,
            Node::Sub(n) => self.exec_binary(&n.inputs, &n.outputs[0].name, env, |a, b| a - b)?,
            Node::Mul(n) => self.exec_binary(&n.inputs, &n.outputs[0].name, env, |a, b| a * b)?,
            Node::Div(n) => self.exec_binary(&n.inputs, &n.outputs[0].name, env, |a, b| a / b)?,
            Node::Flatten(n) => {
                let x = self.resolve_arg(&n.inputs[0], env)?;
                let rank = x.shape.len();
                let axis = n.config.axis.min(rank);
                let dim0: usize = if axis == 0 { 1 } else { x.shape[..axis].iter().product() };
                let dim1: usize = x.shape[axis..].iter().product();
                env.insert(
                    n.outputs[0].name.clone(),
                    RuntimeTensor::new(x.data, vec![dim0, dim1])?,
                );
            }
            Node::Identity(n) => {
                let x = self.resolve_arg(&n.inputs[0], env)?;
                env.insert(n.outputs[0].name.clone(), x);
            }
            Node::Gemm(n) => self.exec_gemm(n, env)?,
            Node::Linear(n) => self.exec_linear(n, env)?,
            Node::MatMul(n) => {
                let a = self.resolve_arg(&n.inputs[0], env)?;
                let b = self.resolve_arg(&n.inputs[1], env)?;
                if a.shape.len() != 2 || b.shape.len() != 2 || a.shape[1] != b.shape[0] {
                    return Err(DspError::InvalidConfig(format!(
                        "ONNX MatMul incompatible shapes {:?} and {:?}",
                        a.shape, b.shape
                    )));
                }
                let batch = a.shape[0];
                let k = a.shape[1];
                let out_dim = b.shape[1];
                let b_t = transpose_2d(&b.data, k, out_dim);
                let data = burn_linear_2d(self.target, &a.data, batch, k, &b_t, out_dim, None)?;
                env.insert(
                    n.outputs[0].name.clone(),
                    RuntimeTensor::new(data, vec![batch, out_dim])?,
                );
            }
            Node::Conv1d(n) => self.exec_conv1d(n, env)?,
            Node::GlobalAveragePool(n) => {
                let x = self.resolve_arg(&n.inputs[0], env)?;
                if x.shape.len() != 3 {
                    return Err(DspError::InvalidConfig(format!(
                        "ONNX GlobalAveragePool expects 3D [N, C, T], got {:?}",
                        x.shape
                    )));
                }
                let (b, c, t) = (x.shape[0], x.shape[1], x.shape[2]);
                let mut out = vec![0.0f32; b * c];
                let inv_t = 1.0 / (t.max(1) as f32);
                for bi in 0..b {
                    for ci in 0..c {
                        let start = (bi * c + ci) * t;
                        let sum: f32 = x.data[start..start + t].iter().sum();
                        out[bi * c + ci] = sum * inv_t;
                    }
                }
                env.insert(
                    n.outputs[0].name.clone(),
                    RuntimeTensor::new(out, vec![b, c, 1])?,
                );
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
        inputs: &[onnx_ir::Argument],
        out_name: &str,
        env: &mut HashMap<String, RuntimeTensor>,
        op: fn(f32, f32) -> f32,
    ) -> DspResult<()> {
        let a = self.resolve_arg(&inputs[0], env)?;
        let b = self.resolve_arg(&inputs[1], env)?;
        let data = if a.data.len() == b.data.len() {
            a.data.iter().zip(b.data.iter()).map(|(&x, &y)| op(x, y)).collect()
        } else if b.data.len() == 1 {
            let s = b.data[0];
            a.data.iter().map(|&x| op(x, s)).collect()
        } else if a.data.len() == 1 {
            let s = a.data[0];
            b.data.iter().map(|&y| op(s, y)).collect()
        } else if a.data.len() % b.data.len() == 0 {
            let stride = b.data.len();
            a.data
                .iter()
                .enumerate()
                .map(|(i, &x)| op(x, b.data[i % stride]))
                .collect()
        } else {
            return Err(DspError::InvalidConfig(format!(
                "ONNX binary broadcast unsupported for {:?} and {:?}",
                a.shape, b.shape
            )));
        };
        let shape = if a.numel() >= b.numel() { a.shape } else { b.shape };
        env.insert(out_name.to_string(), RuntimeTensor { data, shape });
        Ok(())
    }

    fn exec_gemm(
        &self,
        n: &GemmNode,
        env: &mut HashMap<String, RuntimeTensor>,
    ) -> DspResult<()> {
        let a = self.resolve_arg(&n.inputs[0], env)?;
        let b = self.resolve_arg(&n.inputs[1], env)?;
        let c = if n.inputs.len() > 2 && !n.inputs[2].is_optional() {
            Some(self.resolve_arg(&n.inputs[2], env)?)
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

        // burn_linear_2d expects weight in [out_features, in_features]
        let (out_dim, k_b, w_out_in) = if n.config.trans_b != 0 {
            (b.shape[0], b.shape[1], b.data)
        } else {
            (
                b.shape[1],
                b.shape[0],
                transpose_2d(&b.data, b.shape[0], b.shape[1]),
            )
        };

        if k_a != k_b {
            return Err(DspError::InvalidConfig(format!(
                "ONNX Gemm inner dimension mismatch: {k_a} vs {k_b}"
            )));
        }

        let mut y = burn_linear_2d(
            self.target,
            &a_row,
            m,
            k_a,
            &w_out_in,
            out_dim,
            None,
        )?;

        if (n.config.alpha - 1.0).abs() > 1e-6 {
            for v in &mut y {
                *v *= n.config.alpha;
            }
        }
        if let Some(bias) = c {
            if n.config.beta != 0.0 {
                for row in y.chunks_exact_mut(out_dim) {
                    for (j, val) in row.iter_mut().enumerate() {
                        *val += n.config.beta * bias.data[j % bias.data.len()];
                    }
                }
            }
        }

        env.insert(
            n.outputs[0].name.clone(),
            RuntimeTensor::new(y, vec![m, out_dim])?,
        );
        Ok(())
    }

    fn exec_linear(
        &self,
        n: &LinearNode,
        env: &mut HashMap<String, RuntimeTensor>,
    ) -> DspResult<()> {
        let x = self.resolve_arg(&n.inputs[0], env)?;
        let w = self.resolve_arg(&n.inputs[1], env)?;
        let b = if n.inputs.len() > 2 && !n.inputs[2].is_optional() {
            Some(self.resolve_arg(&n.inputs[2], env)?)
        } else {
            None
        };

        let in_dim = *x.shape.last().unwrap_or(&1);
        let batch = x.numel() / in_dim.max(1);

        let (out_dim, w_out_in) = if n.config.transpose_weight {
            (
                w.shape[1],
                transpose_2d(&w.data, w.shape[0], w.shape[1]),
            )
        } else {
            (w.shape[0], w.data)
        };

        let y = burn_linear_2d(
            self.target,
            &x.data,
            batch,
            in_dim,
            &w_out_in,
            out_dim,
            b.as_ref().map(|t| t.data.as_slice()),
        )?;

        let mut out_shape = x.shape[..x.shape.len() - 1].to_vec();
        out_shape.push(out_dim);
        env.insert(
            n.outputs[0].name.clone(),
            RuntimeTensor::new(y, out_shape)?,
        );
        Ok(())
    }

    fn exec_conv1d(
        &self,
        n: &Conv1dNode,
        env: &mut HashMap<String, RuntimeTensor>,
    ) -> DspResult<()> {
        let x = self.resolve_arg(&n.inputs[0], env)?;
        let w = self.resolve_arg(&n.inputs[1], env)?;
        let b = if n.inputs.len() > 2 && !n.inputs[2].is_optional() {
            Some(self.resolve_arg(&n.inputs[2], env)?)
        } else {
            None
        };

        if x.shape.len() != 3 || w.shape.len() != 3 {
            return Err(DspError::InvalidConfig(format!(
                "ONNX Conv1d expects 3D input and weight, got {:?} and {:?}",
                x.shape, w.shape
            )));
        }

        let (batch, in_ch, len) = (x.shape[0], x.shape[1], x.shape[2]);
        let (out_ch, _in_per_group, kernel_size) = (w.shape[0], w.shape[1], w.shape[2]);
        let padding = match &n.config.padding {
            PaddingConfig1d::Valid => 0,
            PaddingConfig1d::Explicit(l, _r) => *l,
        };

        let (data, out_len) = burn_conv1d(
            self.target,
            &x.data,
            batch,
            in_ch,
            len,
            &w.data,
            out_ch,
            kernel_size,
            b.as_ref().map(|t| t.data.as_slice()),
            n.config.stride.max(1),
            padding,
            n.config.dilation.max(1),
            n.config.groups.max(1),
        )?;

        env.insert(
            n.outputs[0].name.clone(),
            RuntimeTensor::new(data, vec![batch, out_ch, out_len])?,
        );
        Ok(())
    }
}
