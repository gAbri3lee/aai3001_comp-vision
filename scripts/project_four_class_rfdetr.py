"""Build an inference-only RF-DETR Nano checkpoint from the trusted project weights."""
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import CLASS_NAMES, LEGACY_CLASS_NAMES


def main():
    directory = ROOT / "models/cloud"
    source = torch.load(directory / "rfdetr-nano_v4.pth", map_location="cpu", weights_only=True)
    if source["model_name"] != "RFDETRNano" or source["args"]["class_names"] != list(LEGACY_CLASS_NAMES):
        raise ValueError("Unexpected RF-DETR architecture or source class order")
    state = source["model"].copy()
    # Retain the final background logit after the four foreground classes.
    rows = [LEGACY_CLASS_NAMES.index(name) for name in CLASS_NAMES] + [len(LEGACY_CLASS_NAMES)]
    pattern = r"(?:transformer\.enc_out_class_embed\.\d+|class_embed)\.(?:weight|bias)"
    heads = [key for key in state if re.fullmatch(pattern, key)]
    if len(heads) != 28 or any(state[key].shape[0] != 6 for key in heads):
        raise ValueError("Unexpected encoder or decoder classifier layout")
    for key in heads:
        state[key] = state[key][rows].clone()
    output = dict(model=state, model_name=source["model_name"], rfdetr_version=source["rfdetr_version"],
                  args={**source["args"], "class_names": list(CLASS_NAMES)},
                  model_config={**source["model_config"], "num_classes": len(CLASS_NAMES)})
    torch.save(output, directory / "rfdetr-nano_v4_four_class.pth")


if __name__ == "__main__":
    main()
