"""
Export CerberusSiamese to ONNX for Hailo DFC compilation.

Edit the CFG dict below, then run:
    %run export_onnx.py     # Jupyter
    python export_onnx.py   # terminal
"""

import argparse
import os
import torch
from Cerberus_Siamese import CerberusSiamese

# ── config ────────────────────────────────────────────────────────────────────
CFG = dict(
    snapshot      = "cerberus_epoch17.pth",  # filename inside snapshots/; set to None for random weights
    snapshots_dir = "snapshots",
    output        = "cerberus_core.onnx",
)
# ──────────────────────────────────────────────────────────────────────────────


def export(weights: str | None, output: str) -> None:
    model = CerberusSiamese()

    if weights:
        ckpt = torch.load(weights, map_location="cpu")
        # Training scripts save {"model": state_dict, "optimizer": ..., ...}
        # Support both that format and a bare state_dict.
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # torch.compile() wraps keys with "_orig_mod." — strip it if present
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
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

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        out  = sess.run(None, {"template": z.numpy(), "search": x.numpy()})
        print(f"onnxruntime check passed — output shape: {out[0].shape}")
    except ImportError:
        print("onnxruntime not installed, skipping verification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=None, help="Filename inside snapshots/")
    parser.add_argument("--output",   default=None, help="Output ONNX file path")
    args = parser.parse_args()

    # CLI args override CFG; CFG is the default when running from an IDE / %run
    snapshot = args.snapshot or CFG["snapshot"]
    output   = args.output   or CFG["output"]

    weights = os.path.join(CFG["snapshots_dir"], snapshot) if snapshot else None

    export(weights, output)
