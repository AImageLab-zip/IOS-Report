srun --nodelist=$(hostname) --partition=all_serial --mem=24G --account=grana_maxillo --cpus-per-task=8 --gres=gpu:1 --time 4:00:00 --pty bash
