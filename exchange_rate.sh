
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear

# python -u run_longExp.py \
#   --is_training 1 \
#   --model_id Solar_$seq_len'_'24 \
#   --model $model_name \
#   --data solar \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 24 \
#   --enc_in 8 \
#   --des 'Exp' \
#   --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Solar_$seq_len'_'24.log 

# python -u run_longExp.py \
#   --is_training 1 \
#   --model_id Exchange_$seq_len'_'24 \
#   --model $model_name \
#   --data exchange \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 24 \
#   --enc_in 8 \
#   --des 'Exp' \
#   --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'24.log 

# python -u run_longExp.py \
#   --is_training 1 \
#   --model_id Electricity_$seq_len'_'24 \
#   --model $model_name \
#   --data electricity \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 24 \
#   --enc_in 8 \
#   --des 'Exp' \
#   --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Electricity_$seq_len'_'24.log 

# python -u run_longExp.py \
#   --is_training 1 \
#   --model_id Taxi_$seq_len'_'24 \
#   --model $model_name \
#   --data taxi \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 24 \
#   --enc_in 8 \
#   --des 'Exp' \
#   --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Taxi_$seq_len'_'24.log 

python -u run_longExp.py \
  --is_training 1 \
  --model_id Wiki_$seq_len'_'24 \
  --model $model_name \
  --data wiki \
  --features M \
  --seq_len $seq_len \
  --pred_len 30 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Wiki_$seq_len'_'30.log 
