"""Build embedding-only, block-only, and combined INT8 variants of a Bekko ONNX export."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic


def fold_weight_aliases(model):
    constants = {item.name for item in model.graph.initializer}
    aliases = {}
    remaining = []
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            node.input[i] = aliases.get(name, name)
        if node.op_type == "Identity" and node.input[0] in constants:
            aliases[node.output[0]] = node.input[0]
        else:
            remaining.append(node)
    del model.graph.node[:]
    model.graph.node.extend(remaining)


def quantize_embedding(model):
    weights = [w for w in model.graph.initializer if w.name.endswith("tok_embeddings.weight")]
    if len(weights) != 1:
        raise ValueError("Expected exactly one shared token embedding table")
    weight = weights[0]
    array = numpy_helper.to_array(weight)
    if array.dtype != np.float32 or array.ndim != 2:
        raise ValueError("Expected an FP32 embedding matrix")
    scale = np.maximum(np.max(np.abs(array), axis=1, keepdims=True) / 127, 1e-8).astype(np.float32)
    quantized = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    weight.CopyFrom(numpy_helper.from_array(quantized, weight.name))
    scale_name = weight.name + ".row_scale"
    model.graph.initializer.append(numpy_helper.from_array(scale, scale_name))
    nodes = []
    count = 0
    for node in model.graph.node:
        if weight.name not in node.input:
            nodes.append(node)
            continue
        if node.op_type != "Gather" or node.input[0] != weight.name:
            raise ValueError("Embedding must only be consumed by Gather")
        output = node.output[0]
        node.output[0] = output + ".int8"
        nodes.extend(
            [
                node,
                helper.make_node(
                    "Cast",
                    [node.output[0]],
                    [output + ".float"],
                    name=node.name + "/CastInt8",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Gather",
                    [scale_name, node.input[1]],
                    [output + ".scale"],
                    name=node.name + "/GatherScale",
                    axis=0,
                ),
                helper.make_node(
                    "Mul",
                    [output + ".float", output + ".scale"],
                    [output],
                    name=node.name + "/RestoreScale",
                ),
            ]
        )
        count += 1
    if count != 2:
        raise ValueError(f"Expected prefix and document embedding gathers, found {count}")
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def quantize_block_weights(model, selected):
    names = {n.input[1] for n in model.graph.node if n.name in selected}
    nodes = []
    for weight in list(model.graph.initializer):
        if weight.name not in names:
            continue
        name = weight.name
        values = numpy_helper.to_array(weight)
        scale = np.maximum(np.max(np.abs(values), axis=0) / 127, 1e-8).astype(np.float32)
        quantized = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
        weight.CopyFrom(numpy_helper.from_array(quantized, name + ".int8"))
        model.graph.initializer.append(numpy_helper.from_array(scale, name + ".scale"))
        # Explicit Cast/Mul keeps this a storage-only variant: no quantized
        # matrix operator or Q/DQ fusion may change activation precision.
        nodes.extend(
            [
                helper.make_node(
                    "Cast",
                    [name + ".int8"],
                    [name + ".float"],
                    name=name + "/CastWeight",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Mul", [name + ".float", name + ".scale"], [name], name=name + "/RestoreWeight"
                ),
            ]
        )
    nodes.extend(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing the FP32 export")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--embedding-only",
        action="store_true",
        help="Write model_int8.onnx and its manifest directly to --output",
    )
    args = parser.parse_args()
    original = onnx.load(args.source / "model.onnx")
    original_heads = {
        w.name: numpy_helper.to_array(w)
        for w in original.graph.initializer
        if w.name.startswith("heads.")
    }
    if len(original_heads) != 6:
        raise ValueError("Expected three FP32 heads with weights and biases")
    records = {}
    for name, embedding, blocks in [
        ("embedding-int8", True, "fp32"),
        ("blocks-int8", False, "dynamic"),
        ("embedding-blocks-int8", True, "dynamic"),
        ("blocks-weight-int8", False, "weight_only"),
        ("embedding-blocks-weight-int8", True, "weight_only"),
    ]:
        if args.embedding_only and name != "embedding-int8":
            continue
        target = args.output if args.embedding_only else args.output / name
        target.mkdir(parents=True, exist_ok=True)
        model = onnx.load(args.source / "model.onnx")
        fold_weight_aliases(model)
        if embedding:
            quantize_embedding(model)
        constants = {w.name for w in model.graph.initializer}
        selected = [
            n.name
            for n in model.graph.node
            if n.op_type == "MatMul" and n.input[1] in constants and n.name.startswith("/encoder/")
        ]
        if not selected:
            raise ValueError("No Transformer linear applications found")
        destination = target / ("model_int8.onnx" if args.embedding_only else "model.onnx")
        if blocks == "dynamic":
            quantize_dynamic(
                model,
                destination,
                nodes_to_quantize=selected,
                op_types_to_quantize=["MatMul"],
                per_channel=True,
                weight_type=QuantType.QInt8,
                extra_options={"MatMulConstBOnly": True},
            )
        else:
            if blocks == "weight_only":
                quantize_block_weights(model, selected)
            onnx.save(model, destination)
        result = onnx.load(destination)
        onnx.checker.check_model(result)
        actual = {w.name: numpy_helper.to_array(w) for w in result.graph.initializer}
        for key, expected in original_heads.items():
            value = actual[key]
            if value.shape != expected.shape and value.ndim == 2:
                value = value.T  # ORT rewrites Gemm to MatMul with transposed weights.
            np.testing.assert_array_equal(value, expected)
            assert actual[key].dtype == np.float32
        integer_ops = sum(n.op_type == "MatMulInteger" for n in result.graph.node)
        assert integer_ops == (len(selected) if blocks == "dynamic" else 0)
        for asset in [
            "tokenizer.json",
            "tokenizer_config.json",
            "parity.json",
            "tokenization-parity.json",
        ]:
            if (args.source / asset).resolve() != (target / asset).resolve():
                shutil.copyfile(args.source / asset, target / asset)
        manifest = json.loads((args.source / "manifest.json").read_text())
        manifest["model_file"] = destination.name
        manifest["model_bytes"] = destination.stat().st_size
        manifest["source_model_sha256"] = hashlib.sha256(
            (args.source / "model.onnx").read_bytes()
        ).hexdigest()
        manifest["model_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        manifest["quantization"] = dict(
            embedding="rowwise_symmetric_int8" if embedding else "fp32",
            blocks={
                "dynamic": "dynamic_uint8_activations_per_channel_int8_weights",
                "weight_only": "per_channel_int8_weights_fp32_compute",
                "fp32": "fp32",
            }[blocks],
            heads="fp32",
            integer_matmuls=integer_ops,
        )
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
        records[name] = dict(bytes=destination.stat().st_size, **manifest)
    (args.output / "variants.json").write_text(json.dumps(records, indent=2))
    print(json.dumps({name: item["bytes"] for name, item in records.items()}, indent=2))


if __name__ == "__main__":
    main()
