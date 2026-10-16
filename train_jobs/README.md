# ViT-S 6L seed 2 training restart

On 2026-10-13, Slurm job **1655440** was submitted with
[resume_vits_6L_seed2.sh](resume_vits_6L_seed2.sh). It resumes
`vits_6L_claude_seed2` from checkpoint **94379**, starting at iteration **94380**
and targeting the final `training_142999` teacher checkpoint.

Job **1655440** completed successfully on **2026-10-14 at 02:12:11 UTC**,
with exit code 0 after 12:09:21. It reached iteration 142999 and preserved the
final teacher checkpoint and metrics in the original project run. The missing
linear and segmentation evaluations are now handled by the
[existing sweep workflow](../eval_jobs/README_vits_sweep.md).

The preceding job **1598549** failed during W&B initialization because the
launcher replaced the Python module's `PYTHONPATH`, hiding `platformdirs`.
The repaired launcher prepends the frozen source directory while preserving
module import paths, and checks imports and GPU availability before training.
It also enables `PYTORCH_ALLOC_CONF=expandable_segments:True`; the first
single-GPU test, **1655023**, exposed a full-batch allocation failure that this
setting resolved. `torchrun --standalone` continues to select an available
rendezvous port, avoiding the earlier `EADDRINUSE` failure.

Before submitting the continuation, one-L40S test job **1655287** completed with
exit code **0** in **1 minute 31 seconds**. It loaded the original four-rank model
and optimizer checkpoint, ran 20 compiled training steps (94380–94399) at the
original per-GPU batch size of 224, connected to W&B under a separate test run,
saved teacher and optimizer checkpoints, and validated finite logged losses
and teacher weights. The test driver bounds the loop without shortening the
original 143000-iteration schedules. Its checkpoints remain isolated from the
continuation's restart state.

The job requests one node, four L40S GPUs, 48 CPUs, 256 GiB RAM, and 18 hours under
account `aip-papyan`. It retains the original saved training recipe, seed 2, and
W&B run ID `dinov3_3ece6627c0c389bf`.

Paths for this submission:

- Original run: `/project/aip-papyan/shared/vits_claude_sweep/vits_6L_claude_seed2`
- Frozen source, config, and launchers: `/scratch/yucz/dinov3_results/vits_claude_sweep_resume/submission-retry-Var3QCl7`
- Training output: `/scratch/yucz/dinov3_results/vits_claude_sweep_resume/vits_6L_claude_seed2`
- Scheduler logs: `/scratch/yucz/dinov3_results/vits_claude_sweep_resume/logs/1655440.{out,err}`
- Successful test: `/scratch/yucz/dinov3_results/vits_claude_sweep_resume/tests/1655287`
- Test logs: `/scratch/yucz/dinov3_results/vits_claude_sweep_resume/logs/test-1655287.{out,err}`

The source archive and saved configuration are identical to those frozen for
job 1598549; only the launcher environment changed. Later repository edits
cannot affect this job.

Checkpoint copying and training run on the allocated compute node. Metrics after
checkpoint 94379 are replayed, so they are omitted from the copied metric history.
On successful completion, the launcher copies the final teacher checkpoint,
retained optimizer checkpoints, metrics, and completion logs back to the original
project run. The original recipe and earlier checkpoints remain available;
`resume_<job-id>/` records the resumed configuration and prior metric history.

The [sweep report](../eval_results/vits_claude_sweep/RESULTS.md) incorporates
the additional seed only after its linear and segmentation evaluations pass
the existing protocol and provenance checks. A dependent CPU job refreshes
the statistics, independent verification, and Markdown report automatically.

To inspect the submitted job:

```bash
squeue -j 1655440 -o '%.18i %.9T %.10M %.10l %R'
tail -n 30 /scratch/yucz/dinov3_results/vits_claude_sweep_resume/logs/1655440.out
```

Do not submit another restart while this job is pending or running. If it fails,
inspect the scratch run's latest complete checkpoint before choosing a restart
point; the script's initial staging checkpoint is fixed to 94379.
