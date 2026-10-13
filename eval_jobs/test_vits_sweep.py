"""Meaningful aggregation and protocol regression checks; execute inside Slurm."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import vits_sweep as sweep


class AggregationTests(unittest.TestCase):
    def test_sample_statistics_and_missing_seed(self):
        records = [{"model": "vits_claude", "task": "linear", "training_seed": s,
                    "metrics": {"accuracy": value}} for s, value in enumerate([70, 72, 74])]
        result = sweep.summaries(records)[0]
        self.assertEqual((result["n"], result["mean"], result["sample_std"]), (3, 72, 2))
        self.assertEqual(result["training_seeds"], [0, 1, 2])
        self.assertEqual(result["coverage"], "complete")
        result = sweep.summaries(records[:2])[0]
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["sample_std"], math.sqrt(2))
        self.assertEqual(result["coverage"], "incomplete")
        self.assertIsNone(sweep.summaries(records[:1])[0]["sample_std"])
        self.assertIsNone(sweep.summaries([])[0]["mean"])

    def test_duplicate_seed_rejected(self):
        record = {"model": "vits_claude", "task": "linear", "training_seed": 0, "metrics": {"accuracy": 70}}
        with self.assertRaises(ValueError):
            sweep.summaries([record, record])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(sweep, "ROOT", self.root)
        self.root_patch.start()
        self.config = self.root / "train.yaml"
        self.config.write_text("train: {seed: 0}\n")
        self.checkpoint = self.root / "teacher_checkpoint.pth"
        self.checkpoint.write_bytes(b"synthetic test checkpoint")
        self.row = {"run": "vits_claude_seed0", "model": "vits_claude", "training_seed": 0,
                    "config": str(self.config), "checkpoint": str(self.checkpoint), "complete": True}

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def fixture(self, task="linear", attempt="1"):
        output = self.root / "outputs/vits_claude_seed0" / task / f"attempt_{attempt}"
        output.mkdir(parents=True)
        config = sweep.expected_config(task, self.row, output)
        cfg_name = "eval_config.yaml" if task == "linear" else "segmentation_config.yaml"
        (output / cfg_name).write_text(yaml.safe_dump(config))
        if task == "linear":
            (output / "results-linear.csv").write_text("validation_accuracy,best_classifier\n72.0,classifier_test\n")
            (output / "results_eval_linear.json").write_text("iter: 100100\n")
        else:
            (output / "results-semantic-segmentation.csv").write_text("mIoU,acc\n30.0,40.0\n")
            history = [{"step": 20000, "metrics": {"mIoU": 30.0, "acc": 40.0}},
                       {"step": 40000, "metrics": {"mIoU": 29.0, "acc": 41.0}}]
            (output / "validation_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in history))
            (output / "model_best.pth").write_bytes(b"test head")
        metrics, selected = sweep.parse_result(task, output)
        record = {**self.row, "task": task, "eval_seed": sweep.SEEDS[task], "output": str(output),
                  "code_sha256": "test-code", "config_sha256": sweep.sha256(self.config),
                  "checkpoint_identity": sweep.file_id(self.checkpoint), "status": "completed",
                  "metrics": metrics, "selected": selected}
        sweep.write_json(output / "attempt.json", record)
        return output, config

    def test_full_config_and_result_validation(self):
        for task in sweep.TASKS:
            output, config = self.fixture(task)
            sweep.validate_attempt(task, self.row, output, "test-code")
            changed = copy.deepcopy(config)
            changed["seed"] += 1
            with self.assertRaises(ValueError):
                sweep.check_config(task, self.row, changed, output)
            changed = copy.deepcopy(config)
            if task == "linear":
                changed["train"]["learning_rates"] = [0.1]
            else:
                changed["scheduler"]["total_iter"] = 20000
            with self.assertRaises(ValueError):
                sweep.check_config(task, self.row, changed, output)
            with self.assertRaises(ValueError):
                sweep.validate_attempt(task, self.row, output, "different-source")

    def test_duplicate_results_excluded(self):
        self.fixture(attempt="1")
        self.fixture(attempt="2")
        good, problems = sweep.collect(self.row, "linear", "test-code")
        self.assertFalse(good)
        self.assertIn("Duplicate valid", problems[-1]["reason"])

    def test_missing_and_failed_results(self):
        self.assertEqual(sweep.collect(self.row, "linear", "test-code"), ([], []))
        output, _ = self.fixture()
        record = sweep.read_json(output / "attempt.json")
        record.update(status="failed", error="synthetic failure")
        sweep.write_json(output / "attempt.json", record)
        good, problems = sweep.collect(self.row, "linear", "test-code")
        self.assertFalse(good)
        self.assertIn("synthetic failure", problems[0]["reason"])

    def test_malformed_and_early_results(self):
        output, _ = self.fixture()
        (output / "results-linear.csv").write_text("validation_accuracy,best_classifier\nnan,test\n")
        with self.assertRaises(ValueError):
            sweep.parse_result("linear", output)
        (output / "results-linear.csv").write_text("validation_accuracy,best_classifier\n72,test\n73,test\n")
        with self.assertRaises(ValueError):
            sweep.parse_result("linear", output)
        (output / "results-linear.csv").write_text("validation_accuracy,best_classifier\n72,test\n")
        (output / "results_eval_linear.json").write_text("iter: 10009\n")
        with self.assertRaises(ValueError):
            sweep.parse_result("linear", output)

    def test_segmentation_uses_best_miou_step(self):
        output, _ = self.fixture("segmentation")
        metrics, selected = sweep.parse_result("segmentation", output)
        self.assertEqual(metrics["acc"], 40)
        self.assertEqual(selected["validation_step"], 20000)
        (output / "results-semantic-segmentation.csv").write_text("mIoU,acc\n29,41\n")
        with self.assertRaises(ValueError):
            sweep.parse_result("segmentation", output)


def run_tests():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AggregationTests))
    if not result.wasSuccessful():
        raise RuntimeError("Sweep validation tests failed")


if __name__ == "__main__":
    run_tests()
