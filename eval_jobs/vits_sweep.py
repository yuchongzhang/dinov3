#!/usr/bin/env python
"""Slurm-only sweep execution, provenance checks, independent retries, and reports."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import uuid

import yaml

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("SWEEP_ROOT", "/scratch/yucz/dinov3_results/vits_claude_sweep_eval"))
TRAIN_ROOT = Path("/project/aip-papyan/shared/vits_claude_sweep")
IMAGENET = Path("/project/aip-papyan/shared/imagenet-1k/data")
ADE_ARCHIVE = Path("/project/aip-papyan/shared/ADE20K/ADEChallengeData2016.zip")
MODELS = ("vits_claude", "vits_1L_claude", "vits_3L_claude", "vits_5L_claude", "vits_6L_claude")
TASKS = ("linear", "segmentation")
LRS = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 7e-2, .1, .15, .2]
SEEDS = {"linear": 0, "segmentation": 100}
REPORT = REPO / "eval_results/vits_claude_sweep"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    def portable(v):
        if isinstance(v, float) and not math.isfinite(v):
            return ".inf" if v > 0 else "-.inf" if v < 0 else ".nan"
        if isinstance(v, dict):
            return {k: portable(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [portable(x) for x in v]
        return v
    tmp.write_text(json.dumps(portable(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_hash():
    h = hashlib.sha256()
    # Only source code; no research data, credentials, caches, or run directories.
    paths = sorted(list((REPO / "dinov3").rglob("*.py")) +
                   list((REPO / "dinov3").rglob("*.yaml")) +
                   list((REPO / "eval_jobs").glob("sweep*")) + [Path(__file__)])
    for path in paths:
        h.update(str(path.relative_to(REPO)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def code_compatible(stored, current):
    if stored == current:
        return True
    # Exact, audited orchestration-only changes preserve evaluation provenance.
    audit = ROOT / "scheduling_compatibility.json"
    if not audit.is_file():
        return False
    try:
        return any(change["from"] == stored and change["to"] == current
                   for change in read_json(audit)["migrations"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def file_id(path):
    s = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def tail(path, size=256 * 1024):
    try:
        with Path(path).open("rb") as f:
            f.seek(max(0, f.seek(0, 2) - size))
            return f.read().decode(errors="replace")
    except FileNotFoundError:
        return ""


def inventory():
    old = {r["run"]: r for r in read_json(ROOT / "inventory.json")} if (ROOT / "inventory.json").exists() else {}
    rows = []
    for model in MODELS:
        for seed in range(3):
            name = f"{model}_seed{seed}"
            run = TRAIN_ROOT / name
            checkpoint = run / "eval/training_142999/teacher_checkpoint.pth"
            config = run / "config.yaml"
            log_path = run / "logs/log.txt"
            log = tail(log_path)
            # A resumed torchrun job writes scheduler stdout instead of replacing
            # the original training log. Require both completion markers together.
            if "[142999/143000]" not in log or "Job completed successfully" not in log:
                for candidate in sorted(run.glob("*_0_log.out")):
                    resumed_log = tail(candidate)
                    if "[142999/143000]" in resumed_log and "Job completed successfully" in resumed_log:
                        log_path, log = candidate, resumed_log
                        break
            metric_lines = tail(run / "training_metrics.json").splitlines()
            last_iteration = None
            for line in reversed(metric_lines):
                try:
                    last_iteration = json.loads(line)["iteration"]
                    break
                except (ValueError, KeyError):
                    pass
            evidence = {"last_iteration": last_iteration,
                        "final_checkpoint_exists": checkpoint.is_file(),
                        "final_iteration_in_log": "[142999/143000]" in log,
                        "successful_completion_in_log": "Job completed successfully" in log,
                        "log": str(log_path),
                        "metrics": str(run / "training_metrics.json")}
            complete = all([last_iteration == 142999, checkpoint.is_file(), config.is_file(),
                            evidence["final_iteration_in_log"], evidence["successful_completion_in_log"]])
            saved_seed = yaml.safe_load(config.read_text())["train"]["seed"] if config.exists() else None
            if saved_seed != seed:
                complete = False
                evidence["seed_mismatch"] = saved_seed
            restarts = [int(p.name) for p in (run / "ckpt").glob("*") if p.name.isdigit()]
            row = {"run": name, "model": model, "training_seed": seed,
                   "complete": complete, "config": str(config), "checkpoint": str(checkpoint),
                   "completion_evidence": evidence, "latest_restart": max(restarts, default=None),
                   "latest_restart_checkpoint": str(run / "ckpt" / str(max(restarts))) if restarts else None,
                   "evaluations": old.get(name, {}).get("evaluations", {})}
            if not complete:
                row["excluded_reason"] = "Training completion evidence is missing or inconsistent."
            rows.append(row)
    write_json(ROOT / "inventory.json", rows)
    return rows


def expected_config(task, row, output, ade_root="LOCAL_ADE20K"):
    from omegaconf import OmegaConf
    from dinov3.eval.setup import ModelConfig
    model = {"config_file": row["config"], "pretrained_weights": row["checkpoint"]}
    if task == "linear":
        from dinov3.eval.linear import LinearEvalConfig
        dataset = f"ImageNetParquet:split={{split}}:root={ROOT / 'staging/imagenet'}"
        overrides = {"model": model, "seed": 0, "output_dir": str(output),
                     "train": {"dataset": dataset.format(split="TRAIN"),
                               "val_dataset": dataset.format(split="VAL"),
                               "num_workers": 12, "batch_size": 128,
                               "epochs": 10, "epoch_length": 10010,
                               "learning_rates": LRS, "n_last_blocks_list": [1],
                               "use_avgpool_list": [True], "resume": False},
                     "eval": {"num_workers": 8}, "transform": {"resize_size": 256, "crop_size": 224}}
        cfg = OmegaConf.merge(OmegaConf.structured(LinearEvalConfig), overrides)
    else:
        from dinov3.eval.segmentation.config import SegmentationConfig
        base = OmegaConf.load(REPO / "dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml")
        overrides = {"model": model, "seed": 100, "output_dir": str(output),
                     "bs": 16, "n_gpus": 1, "num_workers": 6, "model_dtype": "FLOAT32",
                     "datasets": {"root": str(ade_root)}, "eval": {"eval_interval": 20000}}
        cfg = OmegaConf.merge(OmegaConf.structured(SegmentationConfig), base, overrides)
    return OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)


def normalize_config(task, config):
    config = json.loads(json.dumps(config))
    config.pop("output_dir", None)
    if task == "segmentation":
        config["datasets"]["root"] = "LOCAL_ADE20K"
    return config


def check_config(task, row, actual, output):
    expected = expected_config(task, row, output)
    if normalize_config(task, actual) != normalize_config(task, expected):
        raise ValueError("Saved evaluation configuration does not match the requested protocol")
    if actual.get("output_dir") != str(output):
        raise ValueError("Saved configuration refers to a different output directory")


def read_csv_result(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected one result row, found {len(rows)}")
    return rows[0]


def parse_result(task, output):
    output = Path(output)
    if task == "linear":
        raw = read_csv_result(output / "results-linear.csv")
        keys = [k for k in raw if k.endswith("_accuracy")]
        if len(keys) != 1 or not raw.get("best_classifier"):
            raise ValueError("Missing validation accuracy or selected classifier")
        # Completion must be at the end of 100100 iterations, not an intermediate validation.
        if "iter: 100100" not in tail(output / "results_eval_linear.json"):
            raise ValueError("No final linear validation at iteration 100100")
        values = {"accuracy": float(raw[keys[0]])}
        selected = {"classifier": raw["best_classifier"], "validation_step": 100100}
    else:
        raw = read_csv_result(output / "results-semantic-segmentation.csv")
        history = [json.loads(line) for line in (output / "validation_metrics.jsonl").read_text().splitlines()]
        if [r["step"] for r in history] != [20000, 40000]:
            raise ValueError("Segmentation validations must be exactly 20000 and 40000")
        best = max(history, key=lambda r: r["metrics"]["mIoU"])
        values = {k: float(v) for k, v in raw.items()}
        if values != best["metrics"] or not (output / "model_best.pth").is_file():
            raise ValueError("Segmentation result does not match best mIoU validation")
        selected = {"validation_step": best["step"], "decoder_checkpoint": str(output / "model_best.pth")}
    if not values or any(not math.isfinite(v) or not 0 <= v <= 100 for v in values.values()):
        raise ValueError("Malformed percentage metric")
    return values, selected


def validate_attempt(task, row, output, current_code):
    record = read_json(output / "attempt.json")
    if record.get("status") != "completed":
        raise ValueError(record.get("status", "missing status") + ": " + record.get("error", "No completion marker"))
    if record["run"] != row["run"] or record["task"] != task or record["eval_seed"] != SEEDS[task]:
        raise ValueError("Run/task/seed mismatch")
    if not code_compatible(record["code_sha256"], current_code) or record["config_sha256"] != sha256(row["config"]):
        raise ValueError("Source or training configuration changed")
    if record["checkpoint_identity"] != file_id(row["checkpoint"]):
        raise ValueError("Checkpoint changed")
    cfg_name = "eval_config.yaml" if task == "linear" else "segmentation_config.yaml"
    actual = yaml.safe_load((output / cfg_name).read_text())
    check_config(task, row, actual, output)
    values, selected = parse_result(task, output)
    if record.get("metrics") != values or record.get("selected") != selected:
        raise ValueError("Result and completion marker disagree")
    return record


def collect(row, task, digest):
    good, problems = [], []
    for path in sorted((ROOT / "outputs" / row["run"] / task).glob("attempt_*")):
        try:
            good.append(validate_attempt(task, row, path, digest))
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as e:
            problems.append({"run": row["run"], "task": task, "attempt": str(path), "reason": str(e)})
    if len(good) > 1:
        problems.append({"run": row["run"], "task": task, "reason": "Duplicate valid evaluations; excluded from statistics",
                         "attempts": [r["output"] for r in good]})
        good = []
    return good, problems


def summaries(records):
    result = []
    for model in MODELS:
        for task in TASKS:
            selected = [r for r in records if r["model"] == model and r["task"] == task]
            seeds = [r["training_seed"] for r in selected]
            if len(set(seeds)) != len(seeds):
                raise ValueError("Duplicate training seeds")
            metrics = sorted({k for r in selected for k in r["metrics"]}) or (["accuracy"] if task == "linear" else ["mIoU"])
            for metric in metrics:
                contributing = [r for r in selected if metric in r["metrics"]]
                values = [r["metrics"][metric] for r in contributing]
                result.append({"model": model, "task": task, "metric": metric, "units": "percent",
                               "checkpoint": "training_142999/teacher_checkpoint.pth", "eval_seed": SEEDS[task],
                               "training_seeds": sorted(r["training_seed"] for r in contributing),
                               "n": len(values), "mean": statistics.mean(values) if values else None,
                               "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                               "ddof": 1, "coverage": "complete" if len(values) == 3 else "incomplete"})
    return result


def csv_write(path, rows, fields):
    with Path(path).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k)) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in fields})


def aggregate():
    rows = inventory()
    digest = code_hash()
    job_ids = {jid for row in rows for ev in row["evaluations"].values() for jid in ev.get("job_ids", [])}
    states = {}
    if job_ids:
        accounting = subprocess.run(["sacct", "--noheader", "--parsable2", "-X", "-j", ",".join(sorted(job_ids)),
                                     "--format=JobIDRaw,State,ExitCode"], text=True, capture_output=True)
        if accounting.returncode == 0:
            for line in accounting.stdout.splitlines():
                parts = line.split("|")
                if len(parts) >= 3:
                    states[parts[0]] = {"state": parts[1], "exit_code": parts[2]}
    records, excluded = [], []
    for row in rows:
        for task in TASKS:
            valid, problems = collect(row, task, digest) if row["complete"] else ([], [])
            excluded.extend(problems)
            if valid:
                records.extend(valid)
                status = "completed"
            else:
                status = "excluded" if not row["complete"] else "missing_or_invalid"
                excluded.append({"run": row["run"], "task": task,
                                 "reason": row.get("excluded_reason", "Missing valid evaluation")})
            ev = row["evaluations"].setdefault(task, {})
            submissions = ev.get("submissions", [{"job_id": jid, "array_task_id": ev.get("array_task_id", 0)}
                                                  for jid in ev.get("job_ids", [])])
            accounting_states = {s["job_id"]: states.get(f"{s['job_id']}_{s['array_task_id']}",
                                                         states.get(s["job_id"], {"state": "unknown"}))
                                 for s in submissions}
            ev.update({"status": status, "eval_seed": SEEDS[task], "slurm": accounting_states})
            if not valid and accounting_states:
                excluded.append({"run": row["run"], "task": task, "reason": "No compatible completed result; Slurm accounting",
                                 "slurm": accounting_states})
    write_json(ROOT / "inventory.json", rows)
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "finished_training_runs": sum(r["complete"] for r in rows),
              "valid_evaluations": len(records), "expected_evaluations": sum(r["complete"] for r in rows) * len(TASKS),
              "code_sha256": digest, "per_run": records, "summary": summaries(records), "excluded": excluded}
    dest = ROOT / "reports"
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "summary.json", report)
    write_json(dest / "inventory.json", rows)
    csv_write(dest / "summary.csv", report["summary"], ["model", "task", "metric", "units", "checkpoint", "eval_seed",
                                                       "training_seeds", "n", "mean", "sample_std", "ddof", "coverage"])
    flat = [{**r, "metric": k, "value": v, "units": "percent"} for r in records for k, v in r["metrics"].items()]
    csv_write(dest / "per_run.csv", flat, ["run", "model", "training_seed", "task", "eval_seed", "metric", "value",
                                          "units", "checkpoint", "selected", "job_id", "output"])
    csv_write(dest / "inventory.csv", rows, ["run", "model", "training_seed", "complete", "config", "checkpoint",
                                            "completion_evidence", "latest_restart", "latest_restart_checkpoint", "evaluations", "excluded_reason"])
    write_json(dest / "excluded.json", excluded)
    REPORT.mkdir(parents=True, exist_ok=True)
    for path in dest.iterdir():
        if path.suffix in (".json", ".csv"):
            shutil.copy2(path, REPORT / path.name)
    print(json.dumps({"valid_evaluations": len(records), "expected": report["expected_evaluations"], "report": str(REPORT)}), flush=True)
    return report


def stage():
    import numpy as np
    dest = ROOT / "staging"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / ".stage.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for split, count in (("train", 1281167), ("val", 50000)):
            source = IMAGENET / f".cache_{split}"
            target = dest / "imagenet" / f".cache_{split}"
            target.mkdir(parents=True, exist_ok=True)
            for filename in ("labels.npy", "offsets.npy", "images.bin"):
                src, dst = source / filename, target / filename
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    continue
                temp = dst.with_suffix(dst.suffix + ".tmp")
                shutil.copyfile(src, temp)
                temp.replace(dst)
            labels = np.load(target / "labels.npy", mmap_mode="r")
            offsets = np.load(target / "offsets.npy", mmap_mode="r")
            if len(labels) != count or len(offsets) != count + 1 or offsets[-1] != (target / "images.bin").stat().st_size:
                raise ValueError(f"Invalid staged ImageNet cache: {split}")
            if np.any(offsets[1:] <= offsets[:-1]) or labels.min() < 0 or labels.max() >= 1000:
                raise ValueError(f"Invalid ImageNet cache offsets/labels: {split}")
        archive = dest / ADE_ARCHIVE.name
        if not archive.exists() or archive.stat().st_size != ADE_ARCHIVE.stat().st_size:
            shutil.copyfile(ADE_ARCHIVE, archive.with_suffix(".tmp"))
            archive.with_suffix(".tmp").replace(archive)
        # Validate contents without unpacking the many small files onto network storage.
        import zipfile
        # The upstream ZIP omits the split lists required by this repository's loader.
        # Preserve the existing lists used by the reference evaluations.
        with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_DEFLATED) as z:
            for split in ("train", "val"):
                filename = f"ADE20K_object150_{split}.txt"
                member = f"ADEChallengeData2016/{filename}"
                if member not in z.namelist():
                    z.write(ADE_ARCHIVE.parent / "ADEChallengeData2016" / filename, member)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("Corrupt ADE20K archive")
            names = z.namelist()
            counts = {split: sum(n.startswith(f"ADEChallengeData2016/images/{split}/") and n.endswith(".jpg") for n in names)
                      for split in ("training", "validation")}
            if counts != {"training": 20210, "validation": 2000}:
                raise ValueError(f"Invalid ADE20K image counts: {counts}")
            members = set(names)
            for split, count in (("train", 20210), ("val", 2000)):
                listing = z.read(f"ADEChallengeData2016/ADE20K_object150_{split}.txt").decode().strip().splitlines()
                if len(listing) != count or len(set(listing)) != count:
                    raise ValueError(f"Invalid ADE20K {split} split list")
                for image in listing:
                    mask = str(Path(image).with_suffix(".png"))
                    if f"ADEChallengeData2016/images/{image}" not in members or f"ADEChallengeData2016/annotations/{mask}" not in members:
                        raise ValueError(f"Missing image/mask in ADE20K {split} split")
        write_json(dest / "ready.json", {"imagenet_samples": {"train": 1281167, "val": 50000},
                                         "ade_samples": counts, "ade_sha256": sha256(archive)})


def verify():
    import torch
    from dinov3.eval.setup import load_model_and_context, ModelConfig
    from dinov3.run.init import job_context
    rows = inventory()
    if sum(r["complete"] for r in rows) != len(MODELS) * 3:
        raise ValueError("Expected all 15 training runs to be complete")
    records = []
    with job_context(output_dir=str(ROOT / "verification"), seed=0):
        for row in rows:
            if not row["complete"]:
                continue
            output = ROOT / "verification" / row["run"]
            output.mkdir(parents=True, exist_ok=True)
            model, context = load_model_and_context(ModelConfig(config_file=row["config"], pretrained_weights=row["checkpoint"]), str(output))
            checkpoint = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)["teacher"]
            checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}
            state = {k.removeprefix("backbone."): v for k, v in checkpoint.items() if k.startswith("backbone.")}
            if not state:
                state = checkpoint
            # The general eval loader is non-strict; verify every backbone key explicitly.
            model.load_state_dict(state, strict=True)
            model.requires_grad_(False)
            shapes = {}
            with torch.inference_mode():
                for size in (224, 512):
                    with torch.autocast("cuda", dtype=context["autocast_dtype"], enabled=size == 224):
                        patches, cls = model.get_intermediate_layers(torch.zeros(1, 3, size, size, device="cuda"), n=1, return_class_token=True)[0]
                    if not torch.isfinite(patches).all() or not torch.isfinite(cls).all():
                        raise ValueError(f"Nonfinite features: {row['run']}")
                    if patches.shape[-1] != 384 or cls.shape != (1, 384):
                        raise ValueError(f"Unexpected ViT-S features: {row['run']}")
                    shapes[str(size)] = {"patch": list(patches.shape), "cls": list(cls.shape)}
            records.append({"run": row["run"], "checkpoint_identity": file_id(row["checkpoint"]),
                            "checkpoint_sha256": sha256(row["checkpoint"]), "config_sha256": sha256(row["config"]),
                            "features": shapes})
            print(f"VERIFIED {row['run']} {shapes}", flush=True)
            del model, checkpoint, state, patches, cls
            torch.cuda.empty_cache()
    write_json(ROOT / "verification.json", {"code_sha256": code_hash(), "records": records, "job_id": os.environ["SLURM_JOB_ID"]})


def check_verification(rows):
    verification = read_json(ROOT / "verification.json")
    if not code_compatible(verification["code_sha256"], code_hash()):
        raise ValueError("GPU verification is incompatible with current source")
    records = {r["run"]: r for r in verification["records"]}
    for row in rows:
        if not row["complete"]:
            continue
        record = records.get(row["run"])
        if not record or record["checkpoint_identity"] != file_id(row["checkpoint"]) or record["config_sha256"] != sha256(row["config"]):
            raise ValueError(f"Missing or stale checkpoint verification: {row['run']}")
    return records


def run(task, manifest):
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    row = read_json(manifest)[index]
    if not row["complete"]:
        raise ValueError("Refusing evaluation of unfinished training")
    verified = check_verification(inventory())[row["run"]]
    if not (ROOT / "staging/ready.json").is_file():
        raise ValueError("Datasets were not staged successfully")
    digest = code_hash()
    valid, problems = collect(row, task, digest)
    if valid:
        print(f"Compatible completed evaluation exists: {row['run']} {task}", flush=True)
        return
    if any("Duplicate valid" in p["reason"] for p in problems):
        raise ValueError("Duplicate evaluations require review before retrying")
    job_id = os.environ["SLURM_JOB_ID"]
    output = ROOT / "outputs" / row["run"] / task / f"attempt_{job_id}"
    output.mkdir(parents=True, exist_ok=False)
    record = {"run": row["run"], "model": row["model"], "training_seed": row["training_seed"],
              "task": task, "eval_seed": SEEDS[task], "checkpoint": row["checkpoint"],
              "checkpoint_identity": file_id(row["checkpoint"]), "checkpoint_sha256": verified["checkpoint_sha256"],
              "config_sha256": sha256(row["config"]), "code_sha256": digest,
              "job_id": job_id, "output": str(output), "status": "running"}
    write_json(output / "attempt.json", record)
    local = None
    try:
        ade_root = "LOCAL_ADE20K"
        if task == "segmentation":
            import zipfile
            # Vulcan can set SLURM_TMPDIR to /tmp; isolate jobs even on the same node.
            local = Path(os.environ["SLURM_TMPDIR"]) / f"vits_sweep_ade20k_{job_id}_{uuid.uuid4().hex}"
            local.mkdir(mode=0o700, parents=True, exist_ok=False)
            with zipfile.ZipFile(ROOT / "staging" / ADE_ARCHIVE.name) as z:
                z.extractall(local)
            ade_root = str(local / "ADEChallengeData2016")
        config = expected_config(task, row, output, ade_root)
        write_json(output / "requested_config.json", config)
        from omegaconf import OmegaConf
        OmegaConf.save(OmegaConf.create(config), output / "requested_config.yaml")
        os.environ["SWEEP_ATTEMPT"] = str(output)
        launcher = ["torchrun", "--standalone", "--nnodes=1", "--nproc-per-node=1"]
        if task == "linear":
            cmd = launcher + ["-m", "dinov3.eval.linear", f"config_file={output / 'requested_config.yaml'}", f"output_dir={output}"]
        else:
            cmd = launcher + [str(REPO / "eval_jobs/sweep_segmentation.py"), f"config={output / 'requested_config.yaml'}", f"output_dir={output}"]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        cfg_name = "eval_config.yaml" if task == "linear" else "segmentation_config.yaml"
        check_config(task, row, yaml.safe_load((output / cfg_name).read_text()), output)
        metrics, selected = parse_result(task, output)
        record.update(status="completed", metrics=metrics, selected=selected)
    except BaseException as e:
        record.update(status="failed", error=str(e))
        write_json(output / "attempt.json", record)
        raise
    finally:
        if local is not None:
            shutil.rmtree(local)
    write_json(output / "attempt.json", record)


def sbatch(script, args=(), *, dependency=None, task=None, count=None, dry_run=False):
    cmd = ["sbatch", "--parsable", "--export=ALL", f"--chdir={REPO}",
           f"--output={ROOT}/logs/%x_%A_%a.out", f"--error={ROOT}/logs/%x_%A_%a.err"]
    if dependency:
        cmd += [f"--dependency={dependency}", "--kill-on-invalid-dep=yes"]
    if task:
        memory, walltime = {"linear": ("128G", "03:00:00"), "segmentation": ("24G", "04:45:00"),
                            "verify": ("16G", "00:30:00")}[task]
        cmd += ["--account=aip-papyan", "--nodes=1", "--ntasks=1", "--cpus-per-task=16",
                "--gres=gpu:l40s:1", f"--mem={memory}", f"--time={walltime}", f"--job-name=vits_sweep_{task}"]
        if count:
            cmd += [f"--array=0-{count - 1}%2"]
    cmd += [str(REPO / "eval_jobs" / script), *map(str, args)]
    print("SUBMIT " + " ".join(cmd), flush=True)
    if dry_run:
        return "DRY_RUN"
    return subprocess.check_output(cmd, text=True).strip().split(";")[0]


def active_jobs(ids):
    if not ids:
        return set()
    out = subprocess.check_output(["squeue", "--noheader", "--jobs=" + ",".join(sorted(ids)), "--format=%F"], text=True)
    return set(out.split())


def submit(phase, runs=None, tasks=TASKS, dependency=None, dry_run=False):
    rows = inventory()
    digest = code_hash()
    if not dry_run:
        check_verification(rows)
    if phase == "rest":
        for task in TASKS:
            pilot = next(r for r in rows if r["run"] == "vits_claude_seed0")
            good, _ = collect(pilot, task, digest)
            if len(good) != 1:
                raise ValueError(f"Pilot gate failed: {task}")
    requested_ids = {jid for row in rows for ev in row["evaluations"].values() for jid in ev.get("job_ids", [])}
    active = active_jobs(requested_ids) if not dry_run else set()
    jobs = []
    for task in tasks:
        selected = []
        for row in rows:
            if not row["complete"] or (runs and row["run"] not in runs):
                continue
            if phase == "pilot" and row["run"] != "vits_claude_seed0":
                continue
            if phase == "rest" and row["run"] == "vits_claude_seed0":
                continue
            good, problems = collect(row, task, digest)
            if good or any("Duplicate valid" in p["reason"] for p in problems):
                continue
            if active.intersection(row["evaluations"].get(task, {}).get("job_ids", [])):
                continue
            selected.append(row)
        if not selected:
            continue
        manifest = ROOT / "manifests" / f"{phase}_{task}_{uuid.uuid4().hex}.json"
        if not dry_run:
            write_json(manifest, selected)
        job = sbatch("sweep_gpu.sh", (task, manifest), dependency=dependency, task=task, count=len(selected), dry_run=dry_run)
        jobs.append(job)
        if not dry_run:
            for index, row in enumerate(selected):
                ev = row["evaluations"].setdefault(task, {"job_ids": []})
                ev.setdefault("job_ids", []).append(job)
                ev.setdefault("submissions", []).append({"job_id": job, "array_task_id": index})
                ev.update(status="submitted", array_task_id=index, eval_seed=SEEDS[task])
            write_json(ROOT / "inventory.json", rows)
    return jobs


def bootstrap(dry_run=False):
    from test_vits_sweep import run_tests
    run_tests()
    rows = inventory()
    if sum(r["complete"] for r in rows) != len(MODELS) * 3:
        raise ValueError("Inventory does not contain all 15 completed runs")
    # Avoid duplicate orchestration; a failed bootstrap can be retried explicitly.
    existing = read_json(ROOT / "orchestration.json") if (ROOT / "orchestration.json").exists() else {}
    if existing and not dry_run:
        raise ValueError("Workflow already submitted; use submit --phase retry or advance for independent retries")
    stage_id = sbatch("sweep_cpu.sh", ("stage",), dry_run=dry_run)
    verify_id = sbatch("sweep_gpu.sh", ("verify",), task="verify", dry_run=dry_run)
    # A short compute job creates the pilot arrays after both prerequisites succeed.
    launch_id = sbatch("sweep_cpu.sh", ("launch-pilots",), dependency=f"afterok:{stage_id}:{verify_id}", dry_run=dry_run)
    if not dry_run:
        write_json(ROOT / "orchestration.json", {"stage": stage_id, "verify": verify_id, "launch_pilots": launch_id})
        aggregate()


def launch_pilots():
    jobs = submit("pilot")
    next_id = sbatch("sweep_cpu.sh", ("advance",), dependency="afterany:" + ":".join(jobs)) if jobs else sbatch("sweep_cpu.sh", ("advance",))
    state = read_json(ROOT / "orchestration.json")
    state.update(pilots=jobs, advance=next_id)
    write_json(ROOT / "orchestration.json", state)


def advance():
    aggregate()
    jobs = submit("rest")
    final = sbatch("sweep_cpu.sh", ("aggregate",), dependency="afterany:" + ":".join(jobs)) if jobs else None
    state = read_json(ROOT / "orchestration.json")
    state.update(remaining=jobs, aggregate=final)
    write_json(ROOT / "orchestration.json", state)
    aggregate()


def main():
    if "SLURM_JOB_ID" not in os.environ:
        raise SystemExit("Run through eval_jobs/submit_vits_sweep.sh; Python must not run on the login node")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("stage", "verify", "aggregate", "launch-pilots", "advance", "inventory", "test"):
        sub.add_parser(name)
    p = sub.add_parser("bootstrap")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("submit")
    p.add_argument("--phase", choices=("pilot", "rest", "retry"), default="retry")
    p.add_argument("--runs", nargs="+")
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.task, args.manifest)
    elif args.command == "submit":
        with (ROOT / ".submit.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            jobs = submit(args.phase, args.runs, args.tasks, dry_run=args.dry_run)
            if jobs and not args.dry_run:
                sbatch("sweep_cpu.sh", ("aggregate",), dependency="afterany:" + ":".join(jobs))
    elif args.command in ("bootstrap", "launch-pilots", "advance"):
        with (ROOT / ".submit.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.command == "bootstrap":
                bootstrap(args.dry_run)
            else:
                globals()[args.command.replace("-", "_")]()
    elif args.command == "test":
        from test_vits_sweep import run_tests
        run_tests()
    else:
        globals()[args.command.replace("-", "_")]()


if __name__ == "__main__":
    main()
