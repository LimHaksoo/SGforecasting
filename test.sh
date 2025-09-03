seq_len=336
model_name=DLinear
model_id=traffic_$seq_len'_'72_$model_name
pred_len=72

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path traffic.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.05 >"logs/${model_id}_${pred_len}.log"