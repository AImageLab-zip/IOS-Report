srun --nodelist=$(hostname) --partition=all_serial --mem=24G --account=grana_maxillo --gres=gpu:1 --time 4:00:00 --pty bash
