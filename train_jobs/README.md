# ViT-S 6L seed 2 training restart

On 2026-10-12, Slurm job **1598549** was submitted with
[resume_vits_6L_seed2.sh](resume_vits_6L_seed2.sh). It resumes
`vits_6L_claude_seed2` from checkpoint **94379**, starting at iteration **94380**
and targeting the final `training_142999` teacher checkpoint. The prior restart
failed with `EADDRINUSE` on rendezvous port 33279; this launcher uses
`torchrun --standalone` to select an available port.

The job requests one node, four L40S GPUs, 48 CPUs, 256 GiB RAM, and 18 hours under
account `aip-papyan`. It retains the original saved training recipe, seed 2, and
W&B run ID `dinov3_3ece6627c0c389bf`.

Paths for this submission:

- Original run: `/project/aip-papyan/shared/vits_claude_sweep/vits_6L_claude_seed2`
- Frozen source and config: `/scratch/yucz/dinov3_results/vits_claude_sweep_resume/submission-3ubS3EK1`
- Training output: `/scratch/yucz/dinov3_results/vits_claude_sweep_resume/vits_6L_claude_seed2`
- Scheduler logs: `/scratch/yucz/dinov3_results/vits_claude_sweep_resume/logs/1598549.{out,err}`

The source archive includes the current `dinov3/` code, including changes that
were uncommitted at submission. Later repository edits cannot affect this job.
The original saved configuration is also frozen at submission.

Checkpoint copying and training run on the allocated compute node. Metrics after
checkpoint 94379 are replayed, so they are omitted from the copied metric history.
On successful completion, the launcher copies the final teacher checkpoint,
retained optimizer checkpoints, metrics, and completion logs back to the original
project run. The original recipe and earlier checkpoints remain available;
`resume_<job-id>/` records the resumed configuration and prior metric history.

The current [sweep report](../eval_results/vits_claude_sweep/RESULTS.md) contains
14 completed training seeds and excludes both evaluations for this unfinished
seed. This submission does not run linear or segmentation evaluation or change
those statistics. After training completes, update the sweep's explicit seed-2
exclusion and evaluate the final checkpoint before regenerating the report.

To inspect the submitted job:

```bash
squeue -j 1598549 -o '%.18i %.9T %.10M %.10l %R'
tail -n 30 /scratch/yucz/dinov3_results/vits_claude_sweep_resume/logs/1598549.out
```

Do not submit another restart while this job is pending or running. If it fails,
inspect the scratch run's latest complete checkpoint before choosing a restart
point; the script's initial staging checkpoint is fixed to 94379.
