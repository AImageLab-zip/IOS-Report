#!/bin/bash
# Starts debugpy in --wait-for-client mode on port 5678.
# Must be invoked via: srun ... --pty bash launch_debugpy.sh
# (explicit bash needed so /dev/tcp built-in works and background jobs survive --pty)

PYTHON=/homes/llumetti/llumetti_venvs/iosqwen/bin/python
SCRIPT=/work/grana_maxillo/IOS-DraftReport/src/Qwen3-IOS/training/train_intraoralphotos.py
CONFIG=/work/grana_maxillo/IOS-DraftReport/src/Qwen3-IOS/configs/auxiliary_training/qwen3vl_intraoral_photo_finetune.yaml

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/work/grana_maxillo/IOS-DraftReport

echo "[debugpy] Launching Python debugpy on port 5678..."

"$PYTHON" -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    "$SCRIPT" \
    --config "$CONFIG" \
    --train_vision \
    --train_language \
    --run_name qwen3_intraoralphotos_debug &

DEBUGPY_PID=$!

# Poll until port 5678 is bound
for i in $(seq 1 60); do
    if bash -c "echo > /dev/tcp/localhost/5678" 2>/dev/null; then
        break
    fi
    sleep 0.5
done

echo "DEBUGPY_READY"
echo "[debugpy] Port 5678 open — attach from VS Code now."

# Keep the srun session alive; without this, SIGHUP kills the background Python
wait $DEBUGPY_PID
