seq_len=336
model_name=SG_total

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi


for pred_length in 96 192 336 720 
do

  model_id=traffic_$seq_len'_'$model_name
  pred_len=$pred_length

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
    --itr 1 --batch_size 16 --learning_rate 0.001 --device_idx 6 >"logs/${model_id}_${pred_len}.log"

  model_id=weather_$seq_len'_'$model_name

  python -u run_longExp_tfscore.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path weather.csv \
    --model_id $model_id \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --itr 1 --batch_size 16 --learning_rate 0.001 --device_idx 6 >"logs/${model_id}_${pred_len}.log"

  model_id=electricity_$seq_len'_'$model_name

  python -u run_longExp_tfscore.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path electricity.csv \
    --model_id $model_id \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --itr 1 --batch_size 16 --learning_rate 0.001 --device_idx 6 >"logs/${model_id}_${pred_len}.log"
done