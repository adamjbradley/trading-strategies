import time
import onnxruntime as ort
import numpy as np

def benchmark_onnx_model(onnx_path, input_shape=(1, 50, 5), n_runs=100):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # Warm-up
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})

    # Benchmark
    start = time.time()
    for _ in range(n_runs):
        _ = session.run(None, {input_name: dummy_input})
    end = time.time()

    avg_latency_ms = (end - start) / n_runs * 1000
    print(f"ONNX Inference Latency: {avg_latency_ms:.2f} ms")

    return avg_latency_ms
