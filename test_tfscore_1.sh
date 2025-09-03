seq_len=336
model_name=SG_total_woEMA

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi


for learning_rate in 0.1 0.01 0.001 0.0001
do

  model_id=traffic_$seq_len'_'$model_name
  pred_len=96

  python -u run_longExp_tfscore.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path traffic.csv \
    --model_id $model_id \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --itr 1 --batch_size 16 --learning_rate $learning_rate --device_idx 6 >"logs/${model_id}_${pred_len}_${learning_rate}.log"

  # model_id=weather_$seq_len'_'$model_name

  # python -u run_longExp_tfscore.py \
  #   --is_training 1 \
  #   --root_path ./data/ \
  #   --data_path weather.csv \
  #   --model_id $model_id \
  #   --data custom \
  #   --features M \
  #   --seq_len $seq_len \
  #   --pred_len $pred_len \
  #   --des 'Exp' \
  #   --itr 1 --batch_size 16 --learning_rate 0.05 --device_idx 6 >"logs/${model_id}_${pred_len}.log"

  # model_id=electricity_$seq_len'_'$model_name

  # python -u run_longExp_tfscore.py \
  #   --is_training 1 \
  #   --root_path ./data/ \
  #   --data_path electricity.csv \
  #   --model_id $model_id \
  #   --data custom \
  #   --features M \
  #   --seq_len $seq_len \
  #   --pred_len $pred_len \
  #   --des 'Exp' \
  #   --itr 1 --batch_size 16 --learning_rate 0.05 --device_idx 6 >"logs/${model_id}_${pred_len}.log"
done