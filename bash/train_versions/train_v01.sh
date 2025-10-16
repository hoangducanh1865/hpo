PYTHONPATH=. python main.py \
  --train_mode async_hpo \
  --model_name lenet \
  --num_epochs 1 \
  --learning_rate 0.1 \
  --batch_size 256 \
  --num_workers 2 \
  --num_outputs 10 \
  --num_trials 1