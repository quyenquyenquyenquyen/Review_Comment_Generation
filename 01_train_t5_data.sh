#!/bin/bash

# Đường dẫn nguyên gốc checkpoint trên Kaggle
CKPT_SRC=/kaggle/input/checkpoint-1/checkpoint-epoch-1

# Thư mục output của bạn (writable)
OUT=./model/code2review_t5_data_task2/outputs

# Tạo các thư mục cần thiết
mkdir -p model
mkdir -p ./model/code2review_t5_data_task2/cache/
mkdir -p ./model/code2review_t5_data_task2/summary/
mkdir -p ./model/code2review_t5_data_task2/outputs/results

# Copy checkpoint cũ vào đúng vị trí
mkdir -p ${OUT}/checkpoint-epoch-1
cp ${CKPT_SRC}/checkpoint.pt ${OUT}/checkpoint-epoch-1/

# Chạy script huấn luyện
CUDA_VISIBLE_DEVICES=0 python run_gen.py \
    --do_train --do_eval --do_eval_bleu \
    --task refine --sub_task small --model_type codet5 \
    --data_num -1 --num_train_epochs 20 --warmup_steps 500 \
    --learning_rate 5e-5 --patience 3 --beam_size 5 \
    --gradient_accumulation_steps 1 \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --data_dir "/kaggle/input/daataa5/task2_data/t5_data/codet5_format_data" \
    --cache_path ./model/code2review_t5_data_task2/cache/ \
    --output_dir ${OUT} \
    --summary_dir ./model/code2review_t5_data_task2/summary/ \
    --save_last_checkpoints --always_save_model \
    --res_dir ./model/code2review_t5_data_task2/outputs/results \
    --res_fn ./model/code2review_t5_data_task2/outputs/results/summarize_codet5_base.txt \
    --train_batch_size 8 --eval_batch_size 8 \
    --max_source_length 512 --max_target_length 100
