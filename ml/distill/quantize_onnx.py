#!/usr/bin/env python3
"""P3 "Compress": INT8 post-training quantization and graph simplification for ONNX models.

    python ml/distill/quantize_onnx.py model.onnx model.int8.onnx            # dynamic INT8 (weights)
    python ml/distill/quantize_onnx.py model.onnx model.int8.onnx --static calib/  # static, calibrated

Static quantization reads calibration inputs from .npy files named after the model's first input.
Knowledge distillation and structured pruning happen in PyTorch before export (see ml/README.md).

Requires: onnx, onnxruntime; optionally onnxsim.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    import onnx  # type: ignore
    from onnxruntime.quantization import QuantType, quantize_dynamic, quantize_static  # type: ignore
    from onnxruntime.quantization import CalibrationDataReader  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--static", type=Path, help="directory of calibration .npy inputs")
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()

    source = args.model
    if not args.no_simplify:
        try:
            from onnxsim import simplify  # type: ignore

            model, ok = simplify(onnx.load(str(source)))
            if ok:
                source = args.out.with_suffix(".simplified.onnx")
                onnx.save(model, str(source))
                print(f"simplified graph: {source}")
        except ImportError:
            print("onnxsim not installed: skipping graph simplification")

    if args.static is None:
        quantize_dynamic(str(source), str(args.out), weight_type=QuantType.QInt8)
    else:
        import numpy as np  # type: ignore

        input_name = onnx.load(str(source)).graph.input[0].name

        class Reader(CalibrationDataReader):
            def __init__(self) -> None:
                self.files = iter(sorted(args.static.glob("*.npy")))

            def get_next(self):
                path = next(self.files, None)
                return None if path is None else {input_name: np.load(path)}

        quantize_static(str(source), str(args.out), Reader(), weight_type=QuantType.QInt8)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
