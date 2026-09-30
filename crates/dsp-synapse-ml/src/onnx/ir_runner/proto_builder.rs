//! Builder helper for constructing valid serialized `.onnx` (`ModelProto`) byte buffers
//! in memory (used by unit tests and lightweight model exporters without requiring external `.onnx` files).

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
