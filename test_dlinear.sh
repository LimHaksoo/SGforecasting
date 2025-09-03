
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear

for pred_len in 96 192 336 720 
do

  model_id=dlinear_traffic_$seq_len

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

  model_id=dlinear_weather_$seq_len

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path weather.csv \
    --model_id $model_id \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --itr 1 --batch_size 16 >"logs/${model_id}_${pred_len}.log"

  model_id=dlinear_Electricity_$seq_len

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path electricity.csv \
    --model_id $model_id \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 321 \
    --des 'Exp' \
    --itr 1 --batch_size 16  --learning_rate 0.001 >"logs/${model_id}_${pred_len}.log"
done