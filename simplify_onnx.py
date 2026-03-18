"""
Simplify a CerberusSiamese ONNX model with onnx-simplifier.

Edit the CFG dict below, then run:
    %run simplify_onnx.py     # Jupyter
    python simplify_onnx.py   # terminal

Install deps if needed:
    pip install onnx onnxsim
"""

import onnx
import onnxsim

# ── config ────────────────────────────────────────────────────────────────────
CFG = dict(
    input_path  = "cerberus_core.onnx",
    output_path = "cerberus_coreS.onnx",

    # Static input shapes — must match what was used at export time
    input_shapes = {
        "template": [1, 3, 128, 128],
        "search":   [1, 3, 256, 256],
    },

    # How many random inputs to use for numeric equivalence check (check_n)
    # Increase for more confidence; 0 to skip
    check_n = 3,

    # Skip specific onnx optimizers if they cause issues with Hailo DFC
    # e.g. skipped_optimizers = ["fuse_bn_into_conv"]
    skipped_optimizers = None,
)
# ──────────────────────────────────────────────────────────────────────────────


def simplify(
    input_path: str,
    output_path: str,
    input_shapes: dict,
    check_n: int,
    skipped_optimizers,
) -> None:
    print(f"Loading  {input_path} ...")
    model = onnx.load(input_path)
    onnx.checker.check_model(model)

    node_count_before = len(model.graph.node)
    print(f"Nodes before: {node_count_before}")

    model_sim, ok = onnxsim.simplify(
        model,
        overwrite_input_shapes=input_shapes,
        check_n=check_n,
        skipped_optimizers=skipped_optimizers,
    )

    if not ok:
        print("WARNING: onnxsim reported that simplification may not be valid.")
        print("         Saving anyway — inspect carefully before using with Hailo DFC.")
    else:
        print("Simplification validated OK.")

    node_count_after = len(model_sim.graph.node)
    print(f"Nodes after:  {node_count_after}  (reduced by {node_count_before - node_count_after})")

    onnx.save(model_sim, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    simplify(**CFG)
