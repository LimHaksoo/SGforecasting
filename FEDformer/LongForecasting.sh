# cd FEDformer
if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi


# # electricity
# python -u run.py \
#  --is_training 1 \
#  --task_id Solar \
#  --model FEDformer \
#  --data solar \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $preLen \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 137 \
#  --dec_in 137 \
#  --c_out 137 \
#  --des 'Exp' \
#  --itr 1 >../logs/LongForecasting/FEDformer_solar_$pred_len.log

# # exchange
# python -u run.py \
#  --is_training 1 \
#  --task_id Exchange \
#  --model FEDformer \
#  --data exchange \
#  --features S \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $preLen \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 8 \
#  --dec_in 8 \
#  --c_out 8 \
#  --des 'Exp' \
#  --itr 1 >../logs/LongForecasting/FEDformer_exchange_$pred_len.log

# # traffic
# python -u run.py \
#  --is_training 1 \
#  --task_id Electricity \
#  --model FEDformer \
#  --data electricity \
#  --features S \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $preLen \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 370 \
#  --dec_in 370 \
#  --c_out 370 \
#  --des 'Exp' \
#  --itr 1 \
#  --train_epochs 3 >../logs/LongForecasting/FEDformer_electricity_$pred_len.log

# electricity
python -u run.py \
 --is_training 1 \
 --task_id Taxi \
 --model FEDformer \
 --data taxi \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 24 \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1214 \
 --dec_in 1214 \
 --c_out 1214 \
 --des 'Exp' \
 --itr 1

# exchange
python -u run.py \
 --is_training 1 \
 --task_id Wiki \
 --model FEDformer \
 --data wiki \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 30 \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 2000 \
 --dec_in 2000 \
 --c_out 2000 \
 --des 'Exp' \
 --itr 1
