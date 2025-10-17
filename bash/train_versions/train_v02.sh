PYTHONPATH=. python main.py \
  --train_mode multi_fidelity_hpo \
  --model_name lenet \
  --num_epochs 100 \
  --learning_rate 0.1 \
  --batch_size 256 \
  --num_workers 2 \
  --num_outputs 10 \
  --num_trials 20 \
  --max_wallclock_time 600