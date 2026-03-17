"""
Export CerberusSiamese to ONNX for Hailo DFC compilation.

Usage
-----
# untrained weights (architecture test)
python export_onnx.py

# trained weights
python export_onnx.py --weights cerberus.pth --output cerberus.onnx
"""

import argparse
import torch
from Cerberus_Siamese import CerberusSiamese


def export(weights: str | None, output: str) -> None:
    model = CerberusSiamese()

    if weights:
        state = torch.load(weights, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded weights from {weights}")
    else:
        print("No weights provided — exporting with random initialisation")

    model.eval()

    # Fixed batch=1, static shapes — required for Hailo DFC
    z_dummy = torch.zeros(1, 3, 128, 128)
    x_dummy = torch.zeros(1, 3, 256, 256)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (z_dummy, x_dummy),
            output,
            input_names=["template", "search"],
            output_names=["heatmap"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,   # force legacy TorchScript exporter — new dynamo exporter
                            # has an opset 17/18 mismatch bug in the onnxscript inline pass
        )

    print(f"Exported to {output}")
    _verify(output, z_dummy, x_dummy)


def _verify(onnx_path: str, z: torch.Tensor, x: torch.Tensor) -> None:
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        out  = sess.run(None, {"template": z.numpy(), "search": x.numpy()})
        print(f"onnxruntime check passed — output shape: {out[0].shape}")
    except ImportError:
        print("onnxruntime not installed, skipping verification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None,          help="Path to .pth checkpoint (optional)")
    parser.add_argument("--output",  default="cerberus.onnx", help="Output ONNX file path")
    args = parser.parse_args()

    export(args.weights, args.output)
