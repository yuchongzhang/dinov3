"""Preserve the standard trainer while saving the best validation decoder."""
import json
import os
from pathlib import Path

import torch

from dinov3.eval.segmentation import train
from dinov3.eval.segmentation.run import main

original_validate = train.validate


def validate_and_save(*args, **kwargs):
    better, metrics = original_validate(*args, **kwargs)
    # The existing trainer calls validate positionally; retain its exact selection.
    step = kwargs.get("global_step", args[8] if len(args) > 8 else None)
    model = kwargs.get("segmentation_model", args[0] if args else None)
    output = Path(os.environ["SWEEP_ATTEMPT"])
    with (output / "validation_metrics.jsonl").open("a") as f:
        f.write(json.dumps({"step": step, "metrics": metrics, "selected": better}) + "\n")
    if better:
        torch.save({"step": step, "metrics": metrics,
                    "model": {k: v for k, v in model.module.state_dict().items()
                              if "segmentation_model.1" in k}}, output / "model_best.pth")
    return better, metrics


if __name__ == "__main__":
    train.validate = validate_and_save
    raise SystemExit(main())
