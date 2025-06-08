import onnx

model_path = "models/grid_trading_model.onnx"
model = onnx.load(model_path)

print("ONNX Model Inputs:")
for input_tensor in model.graph.input:
    shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            shape.append("None")
    print(f"  {input_tensor.name}: {shape}")

print("\nONNX Model Outputs:")
for output_tensor in model.graph.output:
    shape = []
    for dim in output_tensor.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            shape.append("None")
    print(f"  {output_tensor.name}: {shape}")
