#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _set_workspace(config, gb: float = 2.0):
    bytes_ = int(gb) << 30
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bytes_)
    except Exception:
        config.max_workspace_size = bytes_


class NumpyCalibrator(trt.IInt8EntropyCalibrator2):
    """Simple INT8 calibrator for single-input network."""
    def __init__(self, shape, calib_dir=None, cache="svr.calib", max_batches=20):
        super().__init__()
        self.shape = shape
        self.cache = cache
        self.max_batches = max_batches
        self.idx = 0
        self.files = []
        if calib_dir and os.path.isdir(calib_dir):
            self.files = sorted(glob.glob(os.path.join(calib_dir, "input_*.npy")))[:max_batches]
        self.dev = trt.cuda_malloc(int(np.prod(shape)) * 4)

    def get_batch_size(self):
        return self.shape[0]

    def get_batch(self, names):
        if self.idx >= self.max_batches:
            return None
        if self.files:
            arr = np.load(self.files[self.idx]).astype(np.float32)
        else:
            arr = np.random.rand(*self.shape).astype(np.float32)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, 0)
        trt.memcpy_htod(self.dev, arr)
        self.idx += 1
        return [int(self.dev)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache):
            with open(self.cache, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, data):
        with open(self.cache, "wb") as f:
            f.write(data)


def build_engine(onnx, engine, min_shape, opt_shape, max_shape, fp16=False, int8=False, calib_dir=None, workspace_gb=2.0):
    if not os.path.isfile(onnx):
        raise FileNotFoundError(onnx)

    builder = trt.Builder(TRT_LOGGER)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("ONNX parse error:", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    _set_workspace(config, workspace_gb)

    # One optimization profile for the single input
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)  # "image"
    profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        if not builder.platform_has_fast_int8:
            print("Warning: Platform does not report fast INT8; continuing anyway.")
        config.set_flag(trt.BuilderFlag.INT8)
        calib = NumpyCalibrator(shape=opt_shape, calib_dir=calib_dir,
                                cache=os.path.splitext(engine)[0] + ".calib", max_batches=20)
        config.int8_calibrator = calib

    print(f"[TRT] Building -> {engine}")
    # Prefer modern API
    if hasattr(builder, "build_serialized_network"):
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("build_serialized_network returned None")
        runtime = trt.Runtime(TRT_LOGGER)
        eng = runtime.deserialize_cuda_engine(plan)
    elif hasattr(builder, "build_engine"):
        eng = builder.build_engine(network, config)
    else:
        eng = builder.build_cuda_engine(network)

    if eng is None:
        raise RuntimeError("Engine build failed")

    os.makedirs(os.path.dirname(engine), exist_ok=True)
    with open(engine, "wb") as f:
        f.write(eng.serialize())
    print(f"[TRT] Saved: {engine}")

    # IO summary
    try:
        print("[TRT] IO tensors:")
        for i in range(eng.num_io_tensors):
            name = eng.get_tensor_name(i)
            mode = eng.get_tensor_mode(name)
            shape = eng.get_tensor_shape(name)
            print(f"  - {name} ({'IN' if mode==trt.TensorIOMode.INPUT else 'OUT'}): {shape}")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser("Build TensorRT engine for SVR mask model")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--workspace_gb", type=float, default=2.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib_dir", type=str, default=None)

    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--min_H", type=int, default=224)
    ap.add_argument("--opt_H", type=int, default=224)
    ap.add_argument("--max_H", type=int, default=224)
    ap.add_argument("--min_W", type=int, default=224)
    ap.add_argument("--opt_W", type=int, default=224)
    ap.add_argument("--max_W", type=int, default=224)
    a = ap.parse_args()

    B, C = a.batch, a.channels
    min_shape = (B, C, a.min_H, a.min_W)
    opt_shape = (B, C, a.opt_H, a.opt_W)
    max_shape = (B, C, a.max_H, a.max_W)

    build_engine(a.onnx, a.engine, min_shape, opt_shape, max_shape,
                 fp16=a.fp16, int8=a.int8, calib_dir=a.calib_dir,
                 workspace_gb=a.workspace_gb)


if __name__ == "__main__":
    main()
