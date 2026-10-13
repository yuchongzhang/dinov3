#!/bin/bash
# Submit with sbatch, passing a frozen dinov3 source archive and saved run config.
#SBATCH --job-name=vits-6L-seed2-resume
#SBATCH --account=aip-papyan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=18:00:00

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SCRATCH:?SCRATCH must be set}"
: "${SLURM_TMPDIR:?SLURM_TMPDIR must be set}"
source_archive=${1:?Pass a frozen dinov3 source archive}
saved_config=${2:?Pass the original saved config.yaml}
original_run=/project/aip-papyan/shared/vits_claude_sweep/vits_6L_claude_seed2
resume_root="$SCRATCH/dinov3_results/vits_claude_sweep_resume"
run_dir="$resume_root/vits_6L_claude_seed2"
checkpoint=94379

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.13.2 arrow/19.0.1
source /project/aip-papyan/yucz/venvs/dinov3_env/bin/activate

mkdir -p "$run_dir/ckpt" "$SLURM_TMPDIR/source"
# Freeze the code at submission time; later refactoring cannot change this job.
tar -xzf "$source_archive" -C "$SLURM_TMPDIR/source"
cp "$saved_config" "$run_dir/resume_input_config.yaml"

export PYTHONPATH="$SLURM_TMPDIR/source"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/pycache"
export TMPDIR="$SLURM_TMPDIR"
export XDG_CACHE_HOME="$resume_root/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export WANDB_DIR="$run_dir"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_DATA_DIR="$run_dir/wandb-data"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Numeric checkpoint directories are atomically renamed after all shards save.
# Copy the restart state on the compute node, never on the login node.
if [[ ! -f "$run_dir/ckpt/$checkpoint/.metadata" ]]; then
    test -s "$original_run/ckpt/$checkpoint/.metadata"
    for rank in 0 1 2 3; do
        test -s "$original_run/ckpt/$checkpoint/__${rank}_0.distcp"
    done
    stage_checkpoint="$run_dir/ckpt/.stage-${checkpoint}-${SLURM_JOB_ID}"
    mkdir -p "$stage_checkpoint"
    cp -a "$original_run/ckpt/$checkpoint/." "$stage_checkpoint/"
    mv "$stage_checkpoint" "$run_dir/ckpt/$checkpoint"
    # Discard metrics after the checkpoint: those iterations will be replayed.
    awk -F '"iteration": |,' -v last="$checkpoint" '$2 <= last' \
        "$original_run/training_metrics.json" > "$run_dir/training_metrics.json"
fi

# Preserve the original W&B identity despite moving job output to scratch.
# --standalone selects a free rendezvous port, avoiding the failed fixed port.
cd "$SLURM_TMPDIR/source"
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    -m dinov3.train.train \
    --config-file "$run_dir/resume_input_config.yaml" \
    --output-dir "$run_dir" --seed 2 \
    train.seed=2 wandb.name=vits_6L_claude_seed2 \
    wandb.id=dinov3_3ece6627c0c389bf

test -s "$run_dir/eval/training_142999/teacher_checkpoint.pth"

# Preserve completed results in project storage for the existing sweep workflow.
# Keep all earlier checkpoints and the original recipe as provenance.
attempt_dir="$original_run/resume_${SLURM_JOB_ID}"
mkdir -p "$attempt_dir" "$original_run/eval"
cp "$run_dir/config.yaml" "$attempt_dir/config.yaml"
cp "$original_run/training_metrics.json" "$attempt_dir/prior_training_metrics.json"
rsync -a "$run_dir/ckpt/" "$original_run/ckpt/"
rsync -a "$run_dir/eval/training_142999/" "$original_run/eval/training_142999/"
cp "$run_dir/training_metrics.json" "$original_run/.training_metrics-${SLURM_JOB_ID}.tmp"
mv "$original_run/.training_metrics-${SLURM_JOB_ID}.tmp" "$original_run/training_metrics.json"
echo "Job completed successfully: vits_6L_claude_seed2 at training_142999"
cp "$resume_root/logs/${SLURM_JOB_ID}.out" "$original_run/${SLURM_JOB_ID}_0_log.out"
cp "$resume_root/logs/${SLURM_JOB_ID}.err" "$original_run/${SLURM_JOB_ID}_0_log.err"
