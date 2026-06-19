#!/bin/bash
# Wrapper that allocates a SLURM GPU node and runs Python with all forwarded args.
# Used as the "python" interpreter in .vscode/launch.json so that F5 runs the
# training script inside an srun job instead of on the login node.
exec srun \
    --nodelist=$(hostname) \
    --partition=all_serial \
    --account=grana_maxillo \
    --mem=24G \
    --cpus-per-task=8 \
    --gres=gpu:1 \
    --time=4:00:00 \
    --pty \
    /homes/llumetti/llumetti_venvs/iosqwen/bin/python "$@"
