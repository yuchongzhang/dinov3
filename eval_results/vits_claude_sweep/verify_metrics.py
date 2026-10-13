"""Independently check reported sweep statistics; run inside Slurm."""
import csv
import json
import os
from pathlib import Path

import numpy as np


def main():
    if "SLURM_JOB_ID" not in os.environ:
        raise SystemExit("Run in a Slurm CPU job")
    root = Path(os.environ["SWEEP_ROOT"])
    reports = root / "reports"
    report = json.loads((reports / "summary.json").read_text())
    assert report["valid_evaluations"] == report["expected_evaluations"] == 28
    assert report["finished_training_runs"] == 14
    records = report["per_run"]
    assert len({(r["run"], r["task"]) for r in records}) == 28
    with (reports / "per_run.csv").open() as stream:
        values = list(csv.DictReader(stream))
    checked = 0
    for summary in report["summary"]:
        rows = [r for r in values if (r["model"], r["task"], r["metric"]) ==
                (summary["model"], summary["task"], summary["metric"])]
        seeds = sorted(int(r["training_seed"]) for r in rows)
        expected_seeds = [0, 1] if summary["model"] == "vits_6L_claude" else [0, 1, 2]
        assert seeds == expected_seeds == summary["training_seeds"]
        assert summary["n"] == len(rows) and summary["ddof"] == 1
        assert summary["units"] == "percent"
        assert {int(r["eval_seed"]) for r in rows} == {summary["eval_seed"]}
        samples = np.array([float(r["value"]) for r in rows], dtype=np.float64)
        assert np.isfinite(samples).all()
        assert np.isclose(samples.mean(), summary["mean"], rtol=0, atol=1e-12)
        assert np.isclose(samples.std(ddof=1), summary["sample_std"], rtol=0, atol=1e-12)
        checked += 1
    for record in records:
        output = Path(record["output"])
        filename = "results-linear.csv" if record["task"] == "linear" else "results-semantic-segmentation.csv"
        with (output / filename).open() as stream:
            raw = list(csv.DictReader(stream))
        assert len(raw) == 1
        if record["task"] == "linear":
            key = next(k for k in raw[0] if k.endswith("_accuracy"))
            assert float(raw[0][key]) == record["metrics"]["accuracy"]
            assert raw[0]["best_classifier"] == record["selected"]["classifier"]
            assert record["eval_seed"] == 0
        else:
            history = [json.loads(line) for line in (output / "validation_metrics.jsonl").read_text().splitlines()]
            assert [h["step"] for h in history] == [20000, 40000]
            best = max(history, key=lambda h: h["metrics"]["mIoU"])
            assert {k: float(v) for k, v in raw[0].items()} == best["metrics"] == record["metrics"]
            assert best["step"] == record["selected"]["validation_step"]
            assert record["eval_seed"] == 100
    checks = {"slurm_job_id": os.environ["SLURM_JOB_ID"], "report_generated_utc": report["generated_utc"],
              "valid_evaluations": len(records), "summary_rows_checked": checked,
              "independent_statistics": "numpy mean and std(ddof=1), absolute tolerance 1e-12",
              "raw_results_match": True, "segmentation_selection": "best mIoU over steps 20000 and 40000",
              "passed": True}
    (reports / "metrics_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks), flush=True)
    for model in dict.fromkeys(r["model"] for r in records):
        for seed in range(3):
            pair = {r["task"]: r for r in records if r["model"] == model and r["training_seed"] == seed}
            if not pair:
                continue
            linear, seg = pair["linear"], pair["segmentation"]
            print(f"{model} seed={seed} accuracy={linear['metrics']['accuracy']:.6f} "
                  f"mIoU={seg['metrics']['mIoU']:.2f} "
                  f"classifier={linear['selected']['classifier']} best_seg_step={seg['selected']['validation_step']}")


if __name__ == "__main__":
    main()
