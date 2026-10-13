# CLAUDE.md

## Environment

Activate before any run or install:
`source /home/yucz/projects/aip-papyan/yucz/venvs/dinov3_env/bin/activate`

If needed, run `module load gcc arrow/21.0.1` before activating the virtual environment.

## Verification

After every code change, verify by writing and running a small standalone script (`verify.py`). Rules:

- **Minimal resources.** Must run in seconds on CPU with tiny/synthetic data. Never launch full training, load full datasets, or require GPU.
- **Exercise the real code.** Import the actual classes/functions you changed — don't reimplement logic separately.
- **Use good test cases** Use minimal test cases that actually target the functionality you implemented.
- **Print evidence.** Print shapes, values, or types that prove correctness. Silent success is not sufficient.

Examples: create the real dataloader with synthetic data and print first batch shape; instantiate the model with smallest valid config (depth=1, dim=32) and print output shape of one forward pass; call the modified loss on random tensors and print the result; write real logs and check content.

Delete `verify.py` after it passes.